/*
  Copyright 2024 OPM Contributors.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <opm/simulators/linalg/NeuralCprPolicy.hpp>

#include <opm/common/OpmLog/OpmLog.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace Opm {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

// Binary model layout (export_binary_fallback in train_cpr_policy.py):
//   [num_layers : uint32]
//   [layer_type : uint32]  (3 = Dense)
//   [weights_rows : uint32]  ← input feature count
//   ...
// We read up to the first Dense layer and return weights_rows.
int NeuralCprPolicy::readModelInputSize(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        return CprPolicyFeatures::kNumFeatures;

    auto readU32 = [&](unsigned int& v) {
        f.read(reinterpret_cast<char*>(&v), 4);
        return !f.fail();
    };

    unsigned int num_layers = 0;
    if (!readU32(num_layers) || num_layers == 0)
        return CprPolicyFeatures::kNumFeatures;

    // Scan layers until we hit the first Dense (type=3).
    // Activation layers (type=4) contain only 1 uint32; skip them.
    for (unsigned int i = 0; i < num_layers; ++i) {
        unsigned int layer_type = 0;
        if (!readU32(layer_type))
            break;
        if (layer_type == 3) {  // kDense
            unsigned int weights_rows = 0;
            if (readU32(weights_rows) && weights_rows > 0)
                return static_cast<int>(weights_rows);
            break;
        }
        if (layer_type == 4) {  // kActivation — skip its 1 uint32
            unsigned int dummy = 0;
            readU32(dummy);
        }
        // Scaling/UnScaling layers are not used by our models; stop scanning.
        else break;
    }
    return CprPolicyFeatures::kNumFeatures;
}

NeuralCprPolicy::NeuralCprPolicy(const std::string& model_path)
{
    if (model_path.empty())
        return;

    // NNModel::loadModel uses OPM_ERROR_IF which throws on file-open failure.
    try {
        valid_ = model_.loadModel(model_path);
    } catch (const std::exception& e) {
        OpmLog::warning("NeuralCprPolicy: failed to load model from '"
                        + model_path + "': " + std::string(e.what())
                        + " — using rule-based fallback");
        valid_ = false;
        return;
    }

    if (valid_) {
        model_input_size_ = readModelInputSize(model_path);
        const int nf = CprPolicyFeatures::kNumFeatures;
        OpmLog::info("NeuralCprPolicy: loaded model from " + model_path
                     + " (input=" + std::to_string(model_input_size_)
                     + "/" + std::to_string(nf) + " features"
                     + (model_input_size_ < nf ? ", older model — trailing features zeroed" : "")
                     + ")");
        if (model_input_size_ > nf) {
            OpmLog::warning("NeuralCprPolicy: model expects " + std::to_string(model_input_size_)
                            + " features but binary only provides " + std::to_string(nf)
                            + " — using rule-based fallback");
            valid_ = false;
        }
    }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

bool NeuralCprPolicy::scores(const CprPolicyFeatures& feat,
                             std::vector<float>& out) const
{
    if (!valid_)
        return false;

    auto arr = feat.toArray();
    Opm::ML::Tensor<float> in(model_input_size_);
    static constexpr int active8[] = {0, 1, 2, 3, 4, 6, 7, 8};
    for (int i = 0; i < model_input_size_; ++i)
        in(i) = (model_input_size_ == 8) ? arr[active8[i]] : arr[i];

    Opm::ML::Tensor<float> res;
    if (!model_.apply(in, res) || static_cast<int>(res.data_.size()) < 24)
        return false;
    out.assign(res.data_.begin(), res.data_.end());

    // Forbidden labels get -inf so they are never selected by argmax or
    // included in the softmax normalisation sum.
    for (int i = 0; i < std::min(24, static_cast<int>(out.size())); ++i) {
        if (forbidden_mask_ & (1u << i))
            out[i] = -std::numeric_limits<float>::infinity();
    }
    return true;
}

CprPolicyAction NeuralCprPolicy::predict(const CprPolicyFeatures& feat,
                                          float* out_confidence) const
{
    std::vector<float> out;
    if (!scores(feat, out)) {
        if (out_confidence) *out_confidence = 1.0f;
        return ruleBasedPredict(feat);
    }

    if (out.size() >= 768) {
        // Hierarchical model: CPR(12), AMG(12x9), reuse(12x9x6).
        int bestC = 0, bestA = 0, bestR = 0;
        float best = -std::numeric_limits<float>::infinity();
        std::array<int, 3> cands{};
        std::array<float, 12> cs{};
        for (int c = 0; c < 12; ++c) cs[c] = out[c];
        std::array<int, 12> order{}; for (int c=0;c<12;++c) order[c]=c;
        std::partial_sort(order.begin(), order.begin()+3, order.end(),
                          [&](int a,int b){return cs[a]>cs[b];});
        for (int ci=0; ci<3; ++ci) {
            const int c=order[ci];
            std::array<int,9> ao{}; for(int a=0;a<9;++a) ao[a]=a;
            std::partial_sort(ao.begin(),ao.begin()+3,ao.end(),[&](int a,int b){return out[12+c*9+a]>out[12+c*9+b];});
            for(int ai=0;ai<3;++ai) { const int a=ao[ai];
                for(int r=0;r<6;++r) { const float s=out[120+(c*9+a)*6+r];
                    if(s>best){best=s;bestC=c;bestA=a;bestR=r;}
                }
            }
        }
        if (out_confidence) *out_confidence = 1.0f;
        return actionFromReducedLabel(bestC);
    }

    if (out_confidence) {
        // Softmax over allowed logits only, numerically stable.
        const auto* d = out.data();
        float max_l = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < 24; ++i)
            if (std::isfinite(d[i])) max_l = std::max(max_l, d[i]);
        float sum = 0.f;
        for (int i = 0; i < 24; ++i)
            if (std::isfinite(d[i])) sum += std::exp(d[i] - max_l);
        const int best = static_cast<int>(
            std::max_element(d, d + 24) - d);
        *out_confidence = (sum > 0.f) ? std::exp(d[best] - max_l) / sum : 1.0f;
    }

    return decodeLogits(out);
}

int NeuralCprPolicy::chooseAnchored(const CprPolicyFeatures& feat, int anchor,
                                    float min_gain, float* out_gain) const
{
    if (out_gain) *out_gain = 0.0f;
    std::vector<float> out;
    if (isForbidden(anchor) || !scores(feat, out))
        return anchor;

    // Outputs are -log(cost): the gain of the best label over the anchor is
    // log(cost_anchor / cost_best).
    const int best = static_cast<int>(
        std::max_element(out.begin(), out.end()) - out.begin());
    const float gain = out[best] - out[anchor];
    if (out_gain) *out_gain = gain;
    return (std::isfinite(gain) && gain > min_gain) ? best : anchor;
}

std::pair<int, int> NeuralCprPolicy::chooseAmgAnchored(const CprPolicyFeatures& feat,
                                                       float min_gain, float* out_gain) const
{
    if (out_gain) *out_gain = 0.0f;
    if (!valid_)
        return {0, 0};

    std::vector<float> hierarchical;
    if (scores(feat, hierarchical) && hierarchical.size() >= 768) {
        float best = -std::numeric_limits<float>::infinity(); int bc=0, ba=0;
        std::array<int,12> co{}; for(int c=0;c<12;++c) co[c]=c;
        std::partial_sort(co.begin(),co.begin()+3,co.end(),[&](int x,int y){return hierarchical[x]>hierarchical[y];});
        for(int ci=0;ci<3;++ci){int c=co[ci]; std::array<int,9> ao{}; for(int a=0;a<9;++a)ao[a]=a;
            std::partial_sort(ao.begin(),ao.begin()+3,ao.end(),[&](int x,int y){return hierarchical[12+c*9+x]>hierarchical[12+c*9+y];});
            for(int ai=0;ai<3;++ai){int a=ao[ai]; for(int r=0;r<6;++r){float s=hierarchical[120+(c*9+a)*6+r]; if(s>best){best=s;bc=c;ba=a;}}}
        }
        if(out_gain)*out_gain=0.0f; return {ba/3, ba%3};
    }

    // No forbidden-label mask here: the outputs are AMG parameter levels.
    auto arr = feat.toArray();
    Opm::ML::Tensor<float> in(model_input_size_);
    for (int i = 0; i < model_input_size_; ++i)
        in(i) = arr[i];
    Opm::ML::Tensor<float> res;
    const auto nd = static_cast<int>(kAmgDamping.size());
    const auto nt = static_cast<int>(kAmgCoarsenTarget.size());
    if (!model_.apply(in, res) || static_cast<int>(res.data_.size()) < nd + nt)
        return {0, 0};

    // nd * nt outputs: one score per combination (captures interactions);
    // nd + nt outputs: additive per-parameter scores.
    const auto& o = res.data_;
    const bool joint = static_cast<int>(o.size()) >= nd * nt;
    const auto score = [&](int d, int t) { return joint ? o[d * nt + t] : o[d] + o[nd + t]; };
    const float anchor = score(0, 0);
    std::pair<int, int> best{0, 0};
    float bestScore = anchor;
    for (int d = 0; d < nd; ++d) {
        for (int t = 0; t < nt; ++t) {
            const float s = score(d, t);
            if (s > bestScore) {
                bestScore = s;
                best = {d, t};
            }
        }
    }
    const float gain = bestScore - anchor;
    if (out_gain) *out_gain = gain;
    return (std::isfinite(gain) && gain > min_gain) ? best : std::pair<int, int>{0, 0};
}

int NeuralCprPolicy::predictReuseInterval(const CprPolicyFeatures& feat,
                                          int current_interval) const
{
    if (!valid_)
        return current_interval;

    std::vector<float> hierarchical;
    if (scores(feat, hierarchical) && hierarchical.size() >= 768) {
        float best = -std::numeric_limits<float>::infinity(); int br=2;
        std::array<int,12> co{}; for(int c=0;c<12;++c)co[c]=c;
        std::partial_sort(co.begin(),co.begin()+3,co.end(),[&](int x,int y){return hierarchical[x]>hierarchical[y];});
        for(int ci=0;ci<3;++ci){int c=co[ci]; std::array<int,9> ao{}; for(int a=0;a<9;++a)ao[a]=a;
            std::partial_sort(ao.begin(),ao.begin()+3,ao.end(),[&](int x,int y){return hierarchical[12+c*9+x]>hierarchical[12+c*9+y];});
            for(int ai=0;ai<3;++ai){int a=ao[ai]; for(int r=0;r<6;++r){float s=hierarchical[120+(c*9+a)*6+r]; if(s>best){best=s;br=r;}}}
        }
        return kHierarchicalReuseIntervals[br];
    }

    auto arr = feat.toArray();
    Opm::ML::Tensor<float> in(model_input_size_);
    for (int i = 0; i < model_input_size_; ++i)
        in(i) = arr[i];

    Opm::ML::Tensor<float> res;
    if (!model_.apply(in, res))
        return current_interval;

    constexpr int config_outputs = 24;
    const bool legacy_head = res.data_.size() >= config_outputs + kLegacyReuseIntervals.size();
    if (legacy_head) {
        const auto begin = res.data_.begin() + config_outputs;
        const int choice = static_cast<int>(std::max_element(
            begin, begin + kLegacyReuseIntervals.size()) - begin);
        return kLegacyReuseIntervals[choice];
    }
    const int n_intervals = static_cast<int>(kReuseIntervals.size());
    if (static_cast<int>(res.data_.size()) < config_outputs + n_intervals)
        return current_interval; // Backward-compatible with existing models.

    const auto begin = res.data_.begin() + config_outputs;
    const auto end   = begin + n_intervals;
    const int choice = static_cast<int>(std::max_element(begin, end) - begin);
    return kReuseIntervals[choice];
}

CprPolicyAction NeuralCprPolicy::decodeLogits(const std::vector<float>& logits)
{
    // Forbidden labels have been set to -inf before this call.
    const auto best = std::max_element(logits.begin(), logits.begin() + 24);
    return actionFromLabel(static_cast<int>(std::distance(logits.begin(), best)));
}

CprPolicyAction NeuralCprPolicy::actionFromLabel(int idx)
{
    // Joint 3×2×2×2 = 24-way output.
    // Index i encodes: dec = i/8, cprw = (i/4)%2, fine = (i/2)%2, coarse = i%2
    const int dec    = idx / 8;
    const int cprw   = (idx / 4) % 2;
    const int fine   = (idx / 2) % 2;
    const int coarse = idx % 2;

    CprPolicyAction act;
    switch (dec) {
    case 0:  act.weight_type = CprWeightType::QuasiIMPES;        break;
    case 1:  act.weight_type = CprWeightType::TrueIMPES;         break;
    default: act.weight_type = CprWeightType::TrueIMPESAnalytic; break;
    }
    act.use_cprw        = (cprw   == 1);
    act.fine_smoother   = (fine   == 0) ? CprSmoother::ParOverILU0 : CprSmoother::DILU;
    act.coarse_smoother = (coarse == 0) ? CprSmoother::ILU0        : CprSmoother::DILU;
    return act;
}

CprPolicyAction NeuralCprPolicy::actionFromReducedLabel(int idx)
{
    // Reduced 12-label space: weight(3) × fine smoother(2) × coarse smoother(2).
    CprPolicyAction act;
    const int dec = idx / 4;
    act.use_cprw = true;
    act.weight_type = dec == 0 ? CprWeightType::QuasiIMPES
                       : (dec == 1 ? CprWeightType::TrueIMPES
                                   : CprWeightType::TrueIMPESAnalytic);
    act.fine_smoother = ((idx / 2) % 2 == 0) ? CprSmoother::ParOverILU0 : CprSmoother::DILU;
    act.coarse_smoother = (idx % 2 == 0) ? CprSmoother::ILU0 : CprSmoother::DILU;
    return act;
}

CprPolicyAction NeuralCprPolicy::ruleBasedPredict(const CprPolicyFeatures& feat)
{
    CprPolicyAction act;
    act.use_cprw        = true;
    act.coarse_smoother = CprSmoother::ILU0;

    const bool slowConv = (feat.nl_iteration > 0 && feat.nl_residual_reduce > 0.5);
    const bool manyIter = (feat.nl_iteration >= 4);

    if (slowConv || manyIter) {
        // Struggling to converge — use the most robust (expensive) config.
        act.weight_type   = CprWeightType::TrueIMPES;
        act.fine_smoother = CprSmoother::DILU;
    } else if (feat.nl_iteration == 0) {
        // First Newton step: prefer cheap weight computation and smoother setup.
        act.weight_type   = CprWeightType::QuasiIMPES;
        act.fine_smoother = CprSmoother::ParOverILU0;
    } else {
        // Mid-convergence: balanced choice.
        act.weight_type   = CprWeightType::QuasiIMPES;
        act.fine_smoother = CprSmoother::DILU;
    }
    return act;
}

// ---------------------------------------------------------------------------
// PropertyTree helpers
// ---------------------------------------------------------------------------

PropertyTree NeuralCprPolicy::configToPropertyTree(const CprPolicyAction& act,
                                                    double tol, int maxiter)
{
    using namespace std::string_literals;
    PropertyTree prm;

    prm.put("maxiter",   maxiter);
    prm.put("tol",       tol);
    prm.put("verbosity", 0);
    prm.put("solver",    "bicgstab"s);

    prm.put("preconditioner.type",             act.use_cprw ? "cprw"s : "cpr"s);
    // Matches setupCPRW() (the flow default): well weights off.
    prm.put("preconditioner.use_well_weights", "false"s);
    prm.put("preconditioner.add_wells",        act.use_cprw ? "true"s : "false"s);
    prm.put("preconditioner.weight_type",      weightTypeStr(act.weight_type));
    prm.put("preconditioner.pre_smooth",       0);
    prm.put("preconditioner.post_smooth",      1);
    prm.put("preconditioner.finesmoother.type",       fineSmootherStr(act.fine_smoother));
    prm.put("preconditioner.finesmoother.relaxation", 1.0);
    prm.put("preconditioner.verbosity",        0);
    prm.put("preconditioner.coarsesolver.maxiter",   1);
    prm.put("preconditioner.coarsesolver.tol",       1e-1);
    prm.put("preconditioner.coarsesolver.solver",    "loopsolver"s);
    prm.put("preconditioner.coarsesolver.verbosity", 0);

    const std::string amg = "preconditioner.coarsesolver.preconditioner.";
    prm.put(amg + "type",                "amg"s);
    prm.put(amg + "smoother",            coarseSmootherStr(act.coarse_smoother));
    prm.put(amg + "alpha",               0.333333333333);
    prm.put(amg + "relaxation",          1.0);
    prm.put(amg + "iterations",          1);
    prm.put(amg + "coarsenTarget",       1200);
    prm.put(amg + "pre_smooth",          1);
    prm.put(amg + "post_smooth",         1);
    prm.put(amg + "beta",                0.0);
    prm.put(amg + "verbosity",           0);
    prm.put(amg + "maxlevel",            15);
    prm.put(amg + "skip_isolated",       0);
    prm.put(amg + "accumulate",          1);
    prm.put(amg + "prolongationdamping", 1.0);
    prm.put(amg + "maxdistance",         2);
    prm.put(amg + "maxconnectivity",     15);
    prm.put(amg + "maxaggsize",          6);
    prm.put(amg + "minaggsize",          4);
    return prm;
}

void NeuralCprPolicy::updatePropertyTreeInPlace(const CprPolicyAction& act,
                                                 const CprPolicyAction& prev,
                                                 PropertyTree& prm)
{
    if (act.weight_type != prev.weight_type)
        prm.put("preconditioner.weight_type", weightTypeStr(act.weight_type));
    if (act.fine_smoother != prev.fine_smoother)
        prm.put("preconditioner.finesmoother.type", fineSmootherStr(act.fine_smoother));
    if (act.coarse_smoother != prev.coarse_smoother)
        prm.put("preconditioner.coarsesolver.preconditioner.smoother",
                coarseSmootherStr(act.coarse_smoother));
}

// ---------------------------------------------------------------------------
// String helpers
// ---------------------------------------------------------------------------

std::string NeuralCprPolicy::weightTypeStr(CprWeightType w)
{
    switch (w) {
    case CprWeightType::QuasiIMPES:        return "quasiimpes";
    case CprWeightType::TrueIMPES:         return "trueimpes";
    case CprWeightType::TrueIMPESAnalytic: return "trueimpesanalytic";
    }
    return "trueimpes";
}

std::string NeuralCprPolicy::fineSmootherStr(CprSmoother s)
{
    return (s == CprSmoother::DILU) ? "dilu" : "paroverilu0";
}

std::string NeuralCprPolicy::coarseSmootherStr(CprSmoother s)
{
    return (s == CprSmoother::DILU) ? "dilu" : "ilu0";
}

} // namespace Opm
