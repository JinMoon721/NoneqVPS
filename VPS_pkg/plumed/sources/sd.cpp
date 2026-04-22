/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   PLUMED custom CV: SD
   Implements Grifoni–Piccini–Parrinello-like s_d coordinate.

   sd = - sum_{i<m, g(i)!=g(m)} r_im * delta_i * delta_m
   delta_i = W_i - W0_i
   W_i = sum_{H_j} w_i^(j)
   w_i^(j) = exp(-lambda d_ij) / sum_m exp(-lambda d_mj)

   Author: Jin Moon
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "colvar/Colvar.h"
#include "core/ActionRegister.h"
#include "tools/Vector.h"
#include "tools/Tensor.h"
#include <vector>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace PLMD {
namespace colvar {

class SD : public Colvar {
public:
  explicit SD(const ActionOptions& ao);
  static void registerKeywords(Keywords& keys);
  void calculate() override;

private:
  // Input atom lists
  std::vector<AtomNumber> siteAtoms_;
  std::vector<AtomNumber> hydrogenAtoms_;

  // Per-site metadata
  std::vector<int> groupOfSite_;     // length nSites
  std::vector<double> W0_;           // length nSites

  // Parameters
  double lambda_;
  double eps_;

  // Local index mapping into the requested atom list
  std::vector<AtomNumber> allAtoms_;
  std::vector<unsigned> siteLocal_;      // size nSites : local index in allAtoms_
  std::vector<unsigned> hydrogenLocal_;  // size nH     : local index in allAtoms_

  unsigned findOrAppend_(const AtomNumber& a);
};

PLUMED_REGISTER_ACTION(SD, "SD")

void SD::registerKeywords(Keywords& keys) {
  Colvar::registerKeywords(keys);

  keys.add("atoms", "SITES", "List of site atoms i (e.g., water O and solute heavy atoms that define sites).");
  keys.add("atoms", "HYDROGENS", "List of hydrogen atoms j.");

  keys.add("compulsory", "LAMBDA", "Soft Voronoi lambda (units 1/length).");
  keys.add("optional", "EPS", "Small epsilon to avoid division by zero (default 1e-12).");

  keys.add("compulsory", "GROUP", "Integer group label for each site (same length as SITES). Only pairs with different labels contribute.");
  keys.add("compulsory", "W0", "Reference W0_i for each site (same length as SITES).");

//  keys.setValueDescription("scalar", "s_d coordinate");
}

static auto parseIntCSV (const std::string& in) {
  std::string s=in;
  // remove quotes
  for(char& c: s) if(c=='"' || c=='\'') c=' ';
  // split on commas by turning them into spaces
  for(char& c: s) if(c==',' || c=='\t' || c=='\r' || c=='\n') c=' ';
  std::istringstream iss(s);
  std::vector<int> out;
  std::string tok;
  while(iss >> tok) out.push_back(std::stoi(tok));
  return out;
}

static auto parseDoubleCSV (const std::string& in) {
  std::string s=in;
  for(char& c: s) if(c=='"' || c=='\'') c=' ';
  for(char& c: s) if(c==',' || c=='\t' || c=='\r' || c=='\n') c=' ';
  std::istringstream iss(s);
  std::vector<double> out;
  std::string tok;
  while(iss >> tok) out.push_back(std::stod(tok));
  return out;
}

SD::SD(const ActionOptions& ao) : PLUMED_COLVAR_INIT(ao), lambda_(0.0), eps_(1e-12) {
  parseAtomList("SITES", siteAtoms_);
  parseAtomList("HYDROGENS", hydrogenAtoms_);

  parse("LAMBDA", lambda_);
  parse("EPS", eps_);

  // Parse GROUP as a string then split ints
  std::string groupStr;
  parse("GROUP", groupStr);

  std::string w0Str;
  parse("W0", w0Str);


  groupOfSite_ = parseIntCSV(groupStr);
  W0_ = parseDoubleCSV(w0Str);

  const unsigned nSites = siteAtoms_.size();
  const unsigned nH = hydrogenAtoms_.size();

  log.printf("DEBUG nSites = %u\n", (unsigned)siteAtoms_.size());
  log.printf("DEBUG GROUP raw = [%s]\n", groupStr.c_str());
  log.printf("DEBUG W0 raw    = [%s]\n", w0Str.c_str());
  log.printf("DEBUG GROUP parsed count = %u\n", (unsigned)groupOfSite_.size());
  log.printf("DEBUG W0 parsed count    = %u\n", (unsigned)W0_.size());

  if (nSites == 0) error("SITES list is empty.");
  if (nH == 0) error("HYDROGENS list is empty.");

  if (groupOfSite_.size() != nSites) error("GROUP must have same length as SITES.");
  if (W0_.size() != nSites) error("W0 must have same length as SITES.");

  if (lambda_ <= 0.0) error("LAMBDA must be > 0.");

  // Build allAtoms_ and local mappings
  allAtoms_.clear();
  siteLocal_.resize(nSites);
  hydrogenLocal_.resize(nH);

  for (unsigned i = 0; i < nSites; i++) siteLocal_[i] = findOrAppend_(siteAtoms_[i]);
  for (unsigned j = 0; j < nH; j++) hydrogenLocal_[j] = findOrAppend_(hydrogenAtoms_[j]);

  requestAtoms(allAtoms_);
  addValueWithDerivatives(); // tell PLUMED we provide derivatives
  setNotPeriodic();
  checkRead();

}

unsigned SD::findOrAppend_(const AtomNumber& a) {
  for (unsigned k = 0; k < allAtoms_.size(); k++) {
    if (allAtoms_[k] == a) return k;
  }
  allAtoms_.push_back(a);
  return allAtoms_.size() - 1;
}

void SD::calculate() {
  const unsigned nSites = siteAtoms_.size();
  const unsigned nH = hydrogenAtoms_.size();
  const unsigned nAll = allAtoms_.size();

  // Positions in the order of allAtoms_
  const std::vector<Vector>& pos = getPositions();
  const Pbc& pbc = getPbc();

  // Storage:
  // w(i,j), u(i,j), d(i,j)
  std::vector<double> w(nSites * nH, 0.0);
  std::vector<double> d(nSites * nH, 0.0);
  std::vector<Vector> u(nSites * nH); // u_ij

  auto idx = [nH](unsigned i, unsigned j) { return i * nH + j; };

  // ---- Compute w and u via softmax over sites for each hydrogen j
  for (unsigned j = 0; j < nH; j++) {
    const unsigned Hj = hydrogenLocal_[j];
    // compute a_i = exp(-lambda d_ij) and normalization Z_j
    double Z = 0.0;

    // first pass: distances and exp terms
    std::vector<double> a(nSites, 0.0);

    for (unsigned i = 0; i < nSites; i++) {
      const unsigned Si = siteLocal_[i];
      Vector rij = pbc.distance(pos[Hj], pos[Si]); // (R_site - R_H) with MIC
      double dij = rij.modulo();
      if (dij < eps_) dij = eps_;
      d[idx(i, j)] = dij;
      u[idx(i, j)] = rij / dij;

      const double ai = std::exp(-lambda_ * dij);
      a[i] = ai;
      Z += ai;
    }

    if (Z < eps_) Z = eps_;
    for (unsigned i = 0; i < nSites; i++) {
      w[idx(i, j)] = a[i] / Z;
    }
  }

  // ---- W_i and delta_i
  std::vector<double> W(nSites, 0.0);
  for (unsigned i = 0; i < nSites; i++) {
    double sum = 0.0;
    for (unsigned j = 0; j < nH; j++) sum += w[idx(i, j)];
    W[i] = sum;
  }

  std::vector<double> delta(nSites, 0.0);
  for (unsigned i = 0; i < nSites; i++) delta[i] = W[i] - W0_[i];

  // ---- dDelta_dR: (nSites x nAll) vectors
  // Only sites and hydrogens will be nonzero, but we store full nAll for simplicity.
  std::vector<Vector> dDelta_dR(nSites * nAll, Vector(0.0, 0.0, 0.0));
  auto didx = [nAll](unsigned i, unsigned a) { return i * nAll + a; };

  // ubar_j = sum_m w_mj u_mj
  std::vector<Vector> ubar(nH, Vector(0.0, 0.0, 0.0));
  for (unsigned j = 0; j < nH; j++) {
    Vector s(0.0, 0.0, 0.0);
    for (unsigned m = 0; m < nSites; m++) s += w[idx(m, j)] * u[idx(m, j)];
    ubar[j] = s;
  }

  // Site-coordinate contributions:
  // ∂w_i/∂R_i = -λ w_i (1-w_i) u_i
  // ∂w_k/∂R_i = +λ w_k w_i u_i  (k != i)
  for (unsigned i = 0; i < nSites; i++) {
    const unsigned atomI = siteLocal_[i];

    for (unsigned j = 0; j < nH; j++) {
      const double wi = w[idx(i, j)];
      const Vector& ui = u[idx(i, j)];

      // k == i
      dDelta_dR[didx(i, atomI)] += (-lambda_ * wi * (1.0 - wi)) * ui;

      // k != i affects other sites' W_k as well
      for (unsigned k = 0; k < nSites; k++) {
        if (k == i) continue;
        const double wk = w[idx(k, j)];
        dDelta_dR[didx(k, atomI)] += (lambda_ * wk * wi) * ui;
      }
    }
  }

  // Hydrogen-coordinate contributions:
  // ∂w_i/∂R_Hj = λ w_i (u_i - ubar_j)
  for (unsigned j = 0; j < nH; j++) {
    const unsigned atomH = hydrogenLocal_[j];
    for (unsigned i = 0; i < nSites; i++) {
      const double wi = w[idx(i, j)];
      dDelta_dR[didx(i, atomH)] += (lambda_ * wi) * (u[idx(i, j)] - ubar[j]);
    }
  }

  // ---- Now sd and gradient wrt allAtoms_
  double sd = 0.0;
  std::vector<Vector> grad(nAll, Vector(0.0, 0.0, 0.0));

  for (unsigned a = 0; a < nSites; a++) {
    for (unsigned b = a + 1; b < nSites; b++) {
      if (groupOfSite_[a] == groupOfSite_[b]) continue;

      const unsigned atomA = siteLocal_[a];
      const unsigned atomB = siteLocal_[b];

      Vector rabVec = pbc.distance(pos[atomB], pos[atomA]); // R_a - R_b with MIC
      double rab = rabVec.modulo();
      if (rab < eps_) rab = eps_;
      Vector eab = rabVec / rab;

      const double dA = delta[a];
      const double dB = delta[b];

      sd += -rab * dA * dB;

      // distance derivative
      grad[atomA] += -(dA * dB) * eab;
      grad[atomB] += +(dA * dB) * eab;

      // delta-derivative terms: -rab*(dB dδA + dA dδB)
      for (unsigned t = 0; t < nAll; t++) {
        grad[t] += -rab * (dB * dDelta_dR[didx(a, t)] + dA * dDelta_dR[didx(b, t)]);
      }
    }
  }

  // ---- Return value + derivatives
  setValue(sd);

  // Provide atom derivatives in the order of requested atoms (allAtoms_)
  for (unsigned t = 0; t < nAll; t++) {
    setAtomsDerivatives(t, grad[t]);
  }

  // If you run NPT / flexible cell and want correct virial, you'd implement box derivatives.
  // For fixed cell (common in MLIP runs), zero is typically fine:
  setBoxDerivatives(Tensor(0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0));
}



} // namespace colvar
} // namespace PLMD
