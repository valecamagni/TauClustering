import uproot
import awkward as ak
import numpy as np
from tqdm import tqdm 
import mplhep as hep
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker


def delta_phi(a, b):
    return np.mod(a - b + np.pi, 2*np.pi) - np.pi

def deltaR(eta1, phi1, eta2, phi2):
    dphi = delta_phi(phi1, phi2)
    deta = eta1 - eta2
    return np.sqrt(deta*deta + dphi*dphi)


def seeded_cone_njets0(eta, phi, pt, R_seed=0.2, R_cen=0.4, R_clu=0.4):

    N = len(pt)
    seed_mask = np.ones(N, dtype=bool)

    # 1. Seed finding
    for i in range(N):
        if not seed_mask[i]:
            continue
        for j in range(N):
            if j == i:
                continue
            dr = deltaR(eta[i], phi[i], eta[j], phi[j])
            if dr < R_seed:
                if (pt[j] > pt[i]) or (pt[j] == pt[i] and j < i):
                    seed_mask[i] = False
                    break

    seed_idx = np.sort(np.where(seed_mask)[0])
    seed_to_cluster = {old: new for new, old in enumerate(seed_idx)}

    # 2. Centroid calculation
    cent_eta, cent_phi = [], []
    for s in seed_idx:
        drs = deltaR(eta[s], phi[s], eta, phi)
        mask = (drs < R_cen)

        w = pt[mask]
        dEta = eta[mask] - eta[s]
        dPhi = delta_phi(phi[mask], phi[s])

        cent_eta.append(eta[s] + np.sum(w * dEta) / np.sum(w))
        cent_phi.append(phi[s] + np.sum(w * dPhi) / np.sum(w))

    cent_eta = np.array(cent_eta)
    cent_phi = np.array(cent_phi)

    # 3. Assignment
    assign = np.full(N, -1, dtype=int)

    for i in range(N):
        if len(cent_eta) == 0:
            continue
        drs = deltaR(eta[i], phi[i], cent_eta, cent_phi)
        j = np.argmin(drs)
        if drs[j] < R_clu:
            assign[i] = seed_to_cluster[seed_idx[j]]

    # 4. Guarantee seed assignment
    for s in seed_idx:
        assign[s] = seed_to_cluster[s]

    return assign, seed_mask


def seeded_cone_jets_iter(eta, phi, pt, R_clu=0.4, nseeds=16):
    assigns = np.full(len(pt), -1, dtype=int)
    jpt, jeta, jphi = [], [], []
    for iter in range(nseeds):
        pt_masked = np.where(assigns == -1, pt, 0)
        seed = np.argmax(pt_masked)
        if pt_masked[seed] == 0:
            break
        drs = deltaR(eta[seed], phi[seed], eta, phi)
        mask = (drs < R_clu) & (pt_masked > 0)
        w = pt[mask]
        dEta = eta[mask] - eta[seed]
        dPhi = delta_phi(phi[mask], phi[seed])
        sumpt = np.sum(w)
        jpt.append(sumpt)
        jeta.append(eta[seed] + np.sum(w * dEta) / sumpt)
        jphi.append(phi[seed] + np.sum(w * dPhi) / sumpt)
        assigns[mask] = iter
    jets = np.array(jpt),  np.array(jeta), np.array(jphi), np.zeros(len(jpt))
    return (jets, assigns)
            
def antikt_clusters(eta, phi, pt, R_clu=0.4, pt_min=0.0):
    jets, assigns = antikt_jets(eta, phi, pt, R_clu=R_clu, pt_min=pt_min)
    seed_mask = np.zeros(len(pt), dtype=bool) # not  defined for anti-kt
    return assign, seed_mask
    
def antikt_jets(eta, phi, pt, R_clu=0.4, pt_min=0.0):
    import fastjet
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    mass = np.full_like(pt, 0.13957)
    constituents = [ fastjet.PseudoJet(float(x),float(y),float(z),float(e)) for (x,y,z,e) in zip(px,py,pz,np.sqrt(px**2 + py**2 + pz**2 + mass**2)) ]
    for i, c in enumerate(constituents):
        c.set_user_index(i)
        
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, R_clu)
    cluster = fastjet.ClusterSequence(constituents, jetdef)

    N = len(pt)
    assign = np.full(N, -1, dtype=int)
    jets = cluster.inclusive_jets(pt_min)
    NJ = len(jets)
    jpt, jeta, jphi, jmass, jndau = np.zeros(NJ), np.zeros(NJ), np.zeros(NJ), np.zeros(NJ), np.zeros(NJ, dtype=int)
    for i, jet in enumerate(fastjet.sorted_by_pt(jets)):
        #print(f"jet {i}, pt {jet.pt()}")
        jpt[i] = jet.pt()
        jeta[i] = jet.eta()
        jphi[i] = jet.phi()
        jmass[i] = jet.m()
        for dau in jet.constituents():
            #print(f"    daughter pt {dau.pt()}, idx {dau.user_index()}")
            assign[dau.user_index()] = i

    return (jpt, jeta, jphi, jmass), assign

def antikt_clusters(eta, phi, pt, R_clu=0.4, pt_min=0.0):
    jets, assigns = antikt_jets(eta, phi, pt, R_clu=R_clu, pt_min=pt_min)
    seed_mask = np.zeros(len(pt), dtype=bool) # not  defined for anti-kt
    return assign, seed_mask
    
def to_p4(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return np.stack([e, px, py, pz], axis=-1)


def from_p4(p4):
    e, px, py, pz = p4.T
    pt = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    eta = np.arcsinh(pz / np.maximum(pt, 1e-6))
    mass = np.sqrt(np.maximum(e**2 - (px**2 + py**2 + pz**2), 0))
    return pt, eta, phi, mass


def build_clusters(pt, eta, phi, assign, max_daus=16, mass=0.13957):

    clusters = np.unique(assign[assign >= 0])

    jets_pt, jets_eta, jets_phi, jets_mass = [], [], [], []

    for c in clusters:
        mask = (assign == c)

        p_pt  = pt[mask]
        p_eta = eta[mask]
        p_phi = phi[mask]

        order = np.argsort(p_pt)[::-1][:max_daus]
        p_pt  = p_pt[order]
        p_eta = p_eta[order]
        p_phi = p_phi[order]

        p_mass = np.full_like(p_pt, mass)

        p4 = to_p4(p_pt, p_eta, p_phi, p_mass)
        tot = np.sum(p4, axis=0)

        jpt, jeta, jphi, jmass = from_p4(tot.reshape(1,4))

        jets_pt.append(jpt[0])
        jets_eta.append(jeta[0])
        jets_phi.append(jphi[0])
        jets_mass.append(jmass[0])

    return jets_pt, jets_eta, jets_phi, jets_mass


def count_matches(gen_eta, gen_phi, reco_eta, reco_phi, dR=0.3):

    out = []
    thr2 = dR * dR

    for g_eta, g_phi in zip(gen_eta, gen_phi):
        dphi = np.arctan2(np.sin(reco_phi - g_phi),
                          np.cos(reco_phi - g_phi))
        deta = reco_eta - g_eta
        dr2 = deta * deta + dphi * dphi
        out.append(np.sum(dr2 < thr2))

    return out


def score_counts(counts):
    counts = np.array(counts)
    return (counts.mean() - 1.0)**2 + counts.var()


def evaluate_point(sc, fp, R_seed, R_cen, R_clu, n_events=300):

    total_counts = []

    for i in range(n_events):

        if i % 50 == 0:
            print(f"      > event {i}/{n_events}")

        pt  = ak.to_numpy(sc["L1PF_pt"][i])
        eta = ak.to_numpy(sc["L1PF_eta"][i])
        phi = ak.to_numpy(sc["L1PF_phi"][i])

        gen_eta = ak.to_numpy(fp["GenVisTaus_eta"][i])
        gen_phi = ak.to_numpy(fp["GenVisTaus_phi"][i])


        assign, _ = seeded_cone_njets0(
            eta, phi, pt,
            R_seed=R_seed,
            R_cen=R_cen,
            R_clu=R_clu
        )

        _, reco_eta, reco_phi, _ = build_clusters(pt, eta, phi, assign)

        total_counts.extend(
            count_matches(gen_eta, gen_phi, reco_eta, reco_phi)
        )

    return score_counts(total_counts), np.mean(total_counts), np.std(total_counts)



def match_reco_to_gen(chain, reco_key="ScoutPFTaus", gen_key="GenVisTaus", delta_r_threshold=0.3, nnPuppiTauscut = True):
    
    unmatched = []
    matched_pairs = []

    delta_r2_threshold = delta_r_threshold ** 2

    for event_idx, event in enumerate(chain):

        # GEN taus ---
        gen_pt  = np.array(getattr(event, f"{gen_key}_pt"))
        gen_eta = np.array(getattr(event, f"{gen_key}_eta"))
        gen_phi = np.array(getattr(event, f"{gen_key}_phi"))

        # RECO taus ---
        reco_pt  = np.array(getattr(event, f"{reco_key}_pt"))
        reco_eta = np.array(getattr(event, f"{reco_key}_eta"))
        reco_phi = np.array(getattr(event, f"{reco_key}_phi"))
        
        if reco_key == "L1nnPuppiTaus" and nnPuppiTauscut:
            reco_loose = np.array(getattr(event, f"{reco_key}_passLooseNN"))
            valid = (reco_loose > 0)
            reco_pt = reco_pt[valid]
            reco_eta = reco_eta[valid]
            reco_phi = reco_phi[valid]

        used_reco = set()

        for g_idx, (g_pt, g_eta, g_phi) in enumerate(zip(gen_pt, gen_eta, gen_phi)):

            if len(reco_pt) == 0:
                unmatched.append({"type": "gen", "event": event_idx, "gen_index": g_idx})
                continue

            delta_eta = reco_eta - g_eta
            delta_phi = np.arctan2(np.sin(reco_phi - g_phi), np.cos(reco_phi - g_phi))
            delta_r2  = delta_eta**2 + delta_phi**2
            delta_pT  = np.abs(g_pt - reco_pt)

            best_idx  = np.argmin(delta_r2)
            best_dr2  = delta_r2[best_idx]
            best_dpT  = delta_pT[best_idx]

            if best_dr2 < delta_r2_threshold and best_idx not in used_reco:
                matched_pairs.append({
                    "event": event_idx,
                    "genvistau_index": g_idx,
                    "genvistau_pt": g_pt,
                    "genvistau_eta": g_eta,
                    "genvistau_phi": g_phi,
                    "pfjet_index": best_idx,
                    "pfjet_pt": reco_pt[best_idx],
                    "pfjet_eta": reco_eta[best_idx],
                    "pfjet_phi": reco_phi[best_idx],
                    "dr": np.sqrt(best_dr2),
                    "dpt": np.round(best_dpT, 3)
                })
                used_reco.add(best_idx)
            else:
                unmatched.append({"type": "gen", "event": event_idx, "gen_index": g_idx})

        for r_idx in range(len(reco_pt)):
            if r_idx not in used_reco:
                unmatched.append({"type": "reco", "event": event_idx, "reco_index": r_idx})

    return matched_pairs, unmatched



def compute_efficiency_from_matches(matched, chain, gen_key="GenVisTaus",
                                    pt_bins=np.linspace(0, 60, 26),
                                    eta_condition=lambda eta: np.abs(eta) < 1.52):
    pt_bin_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    total_gen_per_bin = np.zeros(len(pt_bins) - 1)
    matched_gen_per_bin = np.zeros(len(pt_bins) - 1)

    for event in chain:
        gen_pt = np.array(getattr(event, f"{gen_key}_pt"))
        gen_eta = np.array(getattr(event, f"{gen_key}_eta"))
        
        valid = eta_condition(gen_eta)
        gen_pt = gen_pt[valid]
        gen_eta = gen_eta[valid]

        bin_indices = np.digitize(gen_pt, pt_bins) - 1
        for idx in bin_indices:
            if 0 <= idx < len(total_gen_per_bin):
                total_gen_per_bin[idx] += 1
                

    for match in matched:
        pt = match["genvistau_pt"]
        eta = match["genvistau_eta"]
        
        if eta_condition(eta):
            bin_idx = np.digitize(pt, pt_bins) - 1
            if 0 <= bin_idx < len(matched_gen_per_bin):
                matched_gen_per_bin[bin_idx] += 1

    efficiency = np.divide(matched_gen_per_bin, total_gen_per_bin,
                           out=np.zeros_like(matched_gen_per_bin),
                           where=total_gen_per_bin > 0)

    errors = np.sqrt(efficiency * (1 - efficiency) / total_gen_per_bin,
                     out=np.zeros_like(matched_gen_per_bin),
                     where=total_gen_per_bin > 0)

    return pt_bin_centers, efficiency, errors, total_gen_per_bin



def plot_efficiencies_single_region(
    efficiencies, pt_bins, region, outputfile,
    true=True, markers=None, mfc=None, colors=None, labs=None
):

    assert markers is not None
    assert mfc is not None
    assert colors is not None
    assert labs is not None

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(12, 13), dpi=300)

    for l, (label, (pt_centers, eff, errors)) in enumerate(efficiencies.items()):
        ax.errorbar(
            pt_centers, eff, yerr=errors,
            fmt=markers[l], color=colors[l], label=labs[l],
            linestyle='', mfc=mfc[l], markersize=14, capsize=8, linewidth=3
        )

    ax.axhline(0.9, color='black', linestyle='--', linewidth=1)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1)

    ax.set_xlabel(r"$p_T^{GEN}$ [GeV]", fontsize=24, loc='right')
    ax.set_ylabel("Efficiency", fontsize=24)
    ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    hep.cms.label(
        ax=ax,
        llabel="Phase-2 Simulation Preliminary",
        rlabel="PU 200 (14 TeV)",
        loc=0
    )

    ax.text(0.03, 0.93, region,
            transform=ax.transAxes,
            fontweight="bold", fontsize=24, style="italic")

    ax.legend()
    plt.tight_layout()
    plt.savefig(outputfile, dpi = 300)
    plt.show()

    
    
def count_reco_in_cone(chain, reco_key="ScoutPFTaus", gen_key="GenVisTaus", delta_r=0.3):
    counts = [] 

    dr2_thr = delta_r * delta_r

    for event in chain:

        gen_eta = np.array(getattr(event, f"{gen_key}_eta"))
        gen_phi = np.array(getattr(event, f"{gen_key}_phi"))

        reco_eta = np.array(getattr(event, f"{reco_key}_eta"))
        reco_phi = np.array(getattr(event, f"{reco_key}_phi"))

        for g_eta, g_phi in zip(gen_eta, gen_phi):

            dphi = np.arctan2(np.sin(reco_phi - g_phi), np.cos(reco_phi - g_phi))
            deta = reco_eta - g_eta
            dr2 = deta*deta + dphi*dphi

            n_in_cone = np.sum(dr2 < dr2_thr)

            counts.append(n_in_cone)

    return np.array(counts)



def wrap_phi(phi):
    return (phi + np.pi) % (2*np.pi) - np.pi



def plot_event_eta_phi_with_cone(
    event, outputfile, 
    gen_key="GenVisTaus",
    reco_keys=("ScoutPFTaus", "L1PFJets"),
    titles=("ScoutPFTaus", "L1PFJets"),
    R=0.3
):

    fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True, dpi = 300)

    # ---------- GEN ----------
    gen_eta = np.array(getattr(event, f"{gen_key}_eta"))
    gen_phi = wrap_phi(np.array(getattr(event, f"{gen_key}_phi")))

    theta = np.linspace(0, 2*np.pi, 200)

    for ax, reco_key, title in zip(axes, reco_keys, titles):

        # GEN taus
        ax.scatter(
            gen_phi, gen_eta,
            s=140, c='red', marker='*',
            label='GenVisTaus', zorder=5
        )

        # ΔR cones
        for eta0, phi0 in zip(gen_eta, gen_phi):
            ax.plot(
                wrap_phi(phi0 + R*np.sin(theta)),
                eta0 + R*np.cos(theta),
                'r--', linewidth=1, zorder=4
            )

        # RECO taus
        reco_eta = np.array(getattr(event, f"{reco_key}_eta"))
        reco_phi = wrap_phi(np.array(getattr(event, f"{reco_key}_phi")))

        ax.scatter(
            reco_phi, reco_eta,
            s=80, c='blue', zorder=3
        )

        ax.set_title(title, fontsize = 20)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-4, 4)
        ax.set_xlabel(r"$\phi$", fontsize = 20)
        ax.grid(True, linestyle='--', alpha=0.5)

        for y in [1.52, -1.52, 2.4, -2.4]:
            ax.axhline(y, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

    axes[0].set_ylabel(r"$\eta$", fontsize = 20)
    axes[0].legend(loc="upper left", fontsize = 12)
    plt.tight_layout()
    plt.savefig(outputfile, dpi = 300)
    plt.show()



def plot_pf_clusters_event(event, outfile,
                           pf_eta_key="L1PF_eta",
                           pf_phi_key="L1PF_phi",
                           pf_pt_key="L1PF_pt",
                           pf_pdgId_key="L1PF_pdgId",
                           pf_seed_key="SC4AlpakaClusters_is_seed",
                           cluster_key="SC4AlpakaClusters_cluster",
                           gen_key="GenVisTaus",
                           reco_key="ScoutPFTaus",
                           R=0.3,
                           title="PF Candidates colored by cluster"):

    # ---------------------------
    # Load data
    # ---------------------------
    pf_eta = np.array(getattr(event, pf_eta_key))
    pf_phi = wrap_phi(np.array(getattr(event, pf_phi_key)))
    pf_pt = np.array(getattr(event, pf_pt_key))
    pf_pdgId = np.array(getattr(event, pf_pdgId_key))
    pf_seed = np.array(getattr(event, pf_seed_key))
    clusters = np.array(getattr(event, cluster_key))

    gen_eta = np.array(getattr(event, f"{gen_key}_eta"))
    gen_phi = wrap_phi(np.array(getattr(event, f"{gen_key}_phi")))
    gen_pt = np.array(getattr(event, f"{gen_key}_pt"))

    reco_eta = np.array(getattr(event, f"{reco_key}_eta"))
    reco_phi = wrap_phi(np.array(getattr(event, f"{reco_key}_phi")))

    # Unique cluster IDs (excluding -1)
    uniq_clusters = np.unique(clusters[clusters >= 0])
    n_clusters = len(uniq_clusters)

    colors = [f"hsl({int(360*i/n_clusters)}, 70%, 50%)" for i in range(n_clusters)]
    cluster_to_color = {cid: colors[i] for i, cid in enumerate(uniq_clusters)}

    fig = go.Figure()

    # ---------------------------
    # PF candidates per cluster
    # ---------------------------
    for cid in uniq_clusters:
        mask = clusters == cid
        fig.add_trace(go.Scatter(
            y=pf_eta[mask],
            x=pf_phi[mask],
            mode="markers",
            marker=dict(size=6, color=cluster_to_color[cid]),
            name=f"Cluster {cid}",
            customdata=np.stack([pf_pt[mask], pf_pdgId[mask], pf_seed[mask]], axis=-1),
            hovertemplate=
                "η = %{x:.3f}<br>"
                "φ = %{y:.3f}<br>"
                "pT = %{customdata[0]:.2f} GeV<br>"
                "pdgId = %{customdata[1]}<br>"
                "seed = %{customdata[2]}<br>"
                "<extra></extra>"
        ))


    # ---------------------------
    # PF without cluster (-1)
    # ---------------------------
    mask_uncl = clusters < 0
    if mask_uncl.sum() > 0:
        fig.add_trace(go.Scatter(
            y=pf_eta[mask_uncl],
            x=pf_phi[mask_uncl],
            mode="markers",
            marker=dict(size=5, color="gray"),
            name="Unclustered",
            customdata=np.stack([pf_pt[mask_uncl], pf_pdgId[mask_uncl], pf_seed[mask_uncl]], axis=-1),
            hovertemplate=
                "η = %{x:.3f}<br>"
                "φ = %{y:.3f}<br>"
                "pT = %{customdata[0]:.2f} GeV<br>"
                "pdgId = %{customdata[1]}<br>"
                "seed = %{customdata[2]}<br>"
                "<extra></extra>"
        ))

    # ---------------------------
    # Gen taus + cone ΔR
    # ---------------------------
    theta = np.linspace(0, 2*np.pi, 300)
    for eta0, phi0, pt0 in zip(gen_eta, gen_phi, gen_pt):

        d_eta = R * np.cos(theta)
        d_phi = R * np.sin(theta)
        circle_eta = eta0 + d_eta
        circle_phi = wrap_phi(phi0 + d_phi)

        # ΔR cone
        fig.add_trace(go.Scatter(
            y=circle_eta,
            x=circle_phi,
            mode="lines",
            line=dict(color="red", dash="dot", width=2),
            name="ΔR=0.3"
        ))

        # Gen tau marker
        fig.add_trace(go.Scatter(
                y=[eta0], x=[phi0],
                mode="markers",
                marker=dict(size=16, color="red", symbol="star"),
                name="GenVisTau",
                customdata=np.array([[pt0]]),
                hovertemplate=
                    "GenVisTau<br>"
                    "η = %{x:.3f}<br>"
                    "φ = %{y:.3f}<br>"
                    "pT = %{customdata[0]:.2f} GeV<br>"
                    "<extra></extra>"
            ))

    # ---------------------------
    # RECO taus
    # ---------------------------
    fig.add_trace(go.Scatter(
        y=reco_eta,
        x=reco_phi,
        mode="markers",
        marker=dict(size=14, color="blue"),
        name=reco_key
    ))

    fig.update_layout(
        title=title,
        yaxis=dict(title="η", range=[-3, 3], scaleanchor="y", scaleratio=1),
        xaxis=dict(title="φ", range=[-np.pi, np.pi]),
        width=900,
        height=900,
        legend=dict(itemsizing='constant')
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.write_html(outfile)
    
    fig.show()
