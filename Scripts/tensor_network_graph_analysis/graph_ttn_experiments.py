import csv
import os
from pathlib import Path

import networkx as nx
import numpy as np
import quimb.tensor as qtn
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

BRANCHES = ("0", "1", "z")
MACHINE_NODES = ("C0", "C1", "Cz")
DISCARDED_WEIGHT = 1e-10


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _bath_node(branch, index):
    return f"B{branch}_{index}"


def build_graphs(chain_length, architecture):
    buffered = architecture != "direct"
    primal = nx.Graph(architecture=architecture, chain_length=chain_length)
    factor = nx.Graph(architecture=architecture, chain_length=chain_length)

    for node in MACHINE_NODES:
        primal.add_node(node, kind="machine")
        factor.add_node(node, kind="machine")
    primal.add_edges_from([("C0", "C1"), ("C0", "Cz"), ("C1", "Cz")], kind="hyperedge_clique")
    factor.add_node("U3", kind="factor")
    factor.add_edges_from(("U3", node) for node in MACHINE_NODES)

    if buffered:
        primal.add_node("F", kind="buffer")
        factor.add_node("F", kind="buffer")
        primal.add_edge("Cz", "F", kind="system_buffer")
        factor.add_edge("Cz", "F", kind="system_buffer")

    for branch, machine in zip(BRANCHES, MACHINE_NODES):
        bath_nodes = [_bath_node(branch, index) for index in range(1, chain_length + 1)]
        for node in bath_nodes:
            primal.add_node(node, kind="bath", branch=branch)
            factor.add_node(node, kind="bath", branch=branch)
        contact = "F" if buffered and branch == "z" else machine
        primal.add_edge(contact, bath_nodes[0], kind="contact")
        factor.add_edge(contact, bath_nodes[0], kind="contact")
        for left, right in zip(bath_nodes[:-1], bath_nodes[1:]):
            primal.add_edge(left, right, kind="bath_chain")
            factor.add_edge(left, right, kind="bath_chain")

    assert nx.is_tree(factor)
    return primal, factor


def _physical_nodes(graph):
    return [node for node, data in graph.nodes(data=True) if data.get("kind") != "factor"]


def _interleaved_order(chain_length, architecture):
    order = list(MACHINE_NODES)
    if architecture != "direct":
        order.append("F")
    for index in range(1, chain_length + 1):
        order.extend(_bath_node(branch, index) for branch in BRANCHES)
    return order


def _cutwidth(graph, order):
    position = {node: index for index, node in enumerate(order)}
    widths = []
    for cut in range(1, len(order)):
        widths.append(
            sum(
                (position[left] < cut) != (position[right] < cut)
                for left, right in graph.edges
            )
        )
    return max(widths, default=0)


def _ordering_score(graph, order):
    position = {node: index for index, node in enumerate(order)}
    spans = [abs(position[left] - position[right]) for left, right in graph.edges]
    return (
        _cutwidth(graph, order),
        max(spans, default=0),
        sum(spans),
    )


def _optimized_order(graph, seed=2026):
    candidates = []
    for start in graph.nodes:
        candidates.append(list(nx.dfs_preorder_nodes(graph, source=start)))
        candidates.append(list(nx.bfs_tree(graph, source=start)))
    candidates.append(list(nx.utils.reverse_cuthill_mckee_ordering(graph)))
    candidates.extend(list(reversed(order)) for order in list(candidates))
    best = min(candidates, key=lambda order: _ordering_score(graph, order))

    rng = np.random.default_rng(seed)
    current = list(best)
    current_score = _ordering_score(graph, current)
    for _ in range(2500):
        left, right = sorted(rng.choice(len(current), size=2, replace=False))
        proposal = list(current)
        proposal[left], proposal[right] = proposal[right], proposal[left]
        score = _ordering_score(graph, proposal)
        if score < current_score:
            current, current_score = proposal, score
    return current


def ordering_metrics(graph, order):
    cutwidth, bandwidth, total_span = _ordering_score(graph, order)
    return {
        "cutwidth": cutwidth,
        "bandwidth": bandwidth,
        "total_edge_span": total_span,
        "log2_edge_boundary_proxy": cutwidth,
    }


def run_graph_scaling(max_chain_length=10):
    rows = []
    for chain_length in range(1, max_chain_length + 1):
        for architecture in ("direct", "passive_buffer", "floquet_buffer"):
            primal, factor = build_graphs(chain_length, architecture)
            orderings = {
                "interleaved_mps": _interleaved_order(chain_length, architecture),
                "graph_optimized_mps": _optimized_order(primal),
            }
            primal_treewidth, _ = nx.approximation.treewidth_min_fill_in(primal)
            factor_treewidth, _ = nx.approximation.treewidth_min_fill_in(factor)
            for ordering, order in orderings.items():
                rows.append(
                    {
                        "architecture": architecture,
                        "chain_length": chain_length,
                        "physical_vertices": primal.number_of_nodes(),
                        "primal_edges": primal.number_of_edges(),
                        "cycle_rank": primal.number_of_edges() - primal.number_of_nodes() + 1,
                        "primal_treewidth_upper_bound": primal_treewidth,
                        "factor_treewidth": factor_treewidth,
                        "ordering": ordering,
                        **ordering_metrics(primal, order),
                    }
                )
            rows.append(
                {
                    "architecture": architecture,
                    "chain_length": chain_length,
                    "physical_vertices": primal.number_of_nodes(),
                    "primal_edges": primal.number_of_edges(),
                    "cycle_rank": primal.number_of_edges() - primal.number_of_nodes() + 1,
                    "primal_treewidth_upper_bound": primal_treewidth,
                    "factor_treewidth": factor_treewidth,
                    "ordering": "factor_tree_ttn",
                    "cutwidth": 1,
                    "bandwidth": 1,
                    "total_edge_span": factor.number_of_edges(),
                    "log2_edge_boundary_proxy": 1,
                }
            )
    return rows


IDENTITY = sparse.csr_matrix(np.eye(2, dtype=complex))
SIGMA_X = sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
SIGMA_Z = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex))
SIGMA_PLUS = sparse.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex))
SIGMA_MINUS = sparse.csr_matrix(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex))


def _many_body_operator(site_operators, total_sites):
    result = sparse.csr_matrix([[1.0 + 0.0j]])
    for site in range(total_sites):
        result = sparse.kron(result, site_operators.get(site, IDENTITY), format="csr")
    return result


def _physical_order(chain_length, architecture):
    nodes = list(MACHINE_NODES)
    if architecture != "direct":
        nodes.append("F")
    for branch in BRANCHES:
        nodes.extend(_bath_node(branch, index) for index in range(1, chain_length + 1))
    return nodes


def build_hamiltonian(chain_length, architecture):
    nodes = _physical_order(chain_length, architecture)
    site = {node: index for index, node in enumerate(nodes)}
    total_sites = len(nodes)
    dimension = 2**total_sites
    hamiltonian = sparse.csr_matrix((dimension, dimension), dtype=complex)

    local_frequencies = {"C0": 2.0, "C1": 1.0, "Cz": 1.0, "F": 1.0}
    for node in nodes:
        if node.startswith("B"):
            chain_index = int(node.split("_")[1])
            frequency = 0.82 + 0.04 * chain_index
        else:
            frequency = local_frequencies[node]
        hamiltonian += 0.5 * frequency * _many_body_operator(
            {site[node]: SIGMA_Z}, total_sites
        )

    exchange_forward = _many_body_operator(
        {
            site["C0"]: SIGMA_PLUS,
            site["C1"]: SIGMA_MINUS,
            site["Cz"]: SIGMA_MINUS,
        },
        total_sites,
    )
    hamiltonian += 0.18 * (exchange_forward + exchange_forward.getH())

    pair_edges = []
    for branch, machine in zip(BRANCHES, MACHINE_NODES):
        if architecture != "direct" and branch == "z":
            pair_edges.append(("Cz", "F", 0.12))
            contact = "F"
        else:
            contact = machine
        pair_edges.append((contact, _bath_node(branch, 1), 0.14))
        for index in range(1, chain_length):
            pair_edges.append(
                (
                    _bath_node(branch, index),
                    _bath_node(branch, index + 1),
                    0.11 / (1.0 + 0.15 * (index - 1)),
                )
            )
    for left, right, coupling in pair_edges:
        hamiltonian += coupling * _many_body_operator(
            {site[left]: SIGMA_X, site[right]: SIGMA_X}, total_sites
        )

    drive_operator = (
        _many_body_operator({site["F"]: SIGMA_X}, total_sites)
        if architecture == "floquet_buffer"
        else sparse.csr_matrix((dimension, dimension), dtype=complex)
    )
    return hamiltonian.tocsr(), drive_operator.tocsr(), nodes


def _initial_state(nodes):
    local_states = []
    for node in nodes:
        if node == "C0":
            local_states.append(np.array([0.0, 1.0], dtype=complex))
        elif node in {"B0_1", "Bz_1"}:
            local_states.append(np.array([np.sqrt(0.8), np.sqrt(0.2)], dtype=complex))
        else:
            local_states.append(np.array([1.0, 0.0], dtype=complex))
    state = np.array([1.0 + 0.0j])
    for local_state in local_states:
        state = np.kron(state, local_state)
    return state / np.linalg.norm(state)


def evolve_state(chain_length, architecture, t_end=4.0, steps=24):
    static_hamiltonian, drive_operator, nodes = build_hamiltonian(chain_length, architecture)
    state = _initial_state(nodes)
    times = np.linspace(0.0, t_end, steps + 1)
    sampled_states = [(times[0], state.copy())]
    for step, (left, right) in enumerate(zip(times[:-1], times[1:]), start=1):
        midpoint = 0.5 * (left + right)
        if architecture == "floquet_buffer":
            hamiltonian = static_hamiltonian + 0.45 * np.cos(1.3 * midpoint) * drive_operator
        else:
            hamiltonian = static_hamiltonian
        state = expm_multiply((-1.0j * (right - left)) * hamiltonian, state)
        state /= np.linalg.norm(state)
        if step % 4 == 0 or step == steps:
            sampled_states.append((right, state.copy()))
    return nodes, sampled_states


def _ordered_state(state, nodes, order):
    axes = [nodes.index(node) for node in order]
    return state.reshape([2] * len(nodes)).transpose(axes).reshape(-1)


def _schmidt_data(state, nodes, left_nodes):
    left_nodes = list(left_nodes)
    if len(left_nodes) > len(nodes) // 2:
        left_nodes = [node for node in nodes if node not in set(left_nodes)]
    right_nodes = [node for node in nodes if node not in set(left_nodes)]
    axes = [nodes.index(node) for node in left_nodes + right_nodes]
    matrix = state.reshape([2] * len(nodes)).transpose(axes).reshape(
        2 ** len(left_nodes), 2 ** len(right_nodes)
    )
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    probabilities = singular_values**2
    probabilities /= probabilities.sum()
    tail = np.concatenate((np.cumsum(probabilities[::-1])[::-1], [0.0]))
    rank = len(probabilities)
    for retained in range(1, len(probabilities) + 1):
        if tail[retained] <= DISCARDED_WEIGHT:
            rank = retained
            break
    nonzero = probabilities[probabilities > 1e-15]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))
    return rank, entropy


def _mps_bond_profile(state, nodes, order):
    ranks = []
    entropies = []
    for cut in range(1, len(order)):
        rank, entropy = _schmidt_data(state, nodes, order[:cut])
        ranks.append(rank)
        entropies.append(entropy)
    bonds = [1] + ranks + [1]
    parameter_count = sum(
        2 * left * right for left, right in zip(bonds[:-1], bonds[1:])
    )
    return ranks, entropies, parameter_count


def _factor_tree_profile(state, nodes, factor_graph):
    ranks = {}
    entropies = []
    for left, right in factor_graph.edges:
        graph = factor_graph.copy()
        graph.remove_edge(left, right)
        component = nx.node_connected_component(graph, left)
        physical_component = [node for node in component if node != "U3"]
        rank, entropy = _schmidt_data(state, nodes, physical_component)
        ranks[frozenset((left, right))] = rank
        entropies.append(entropy)

    parameter_count = 0
    for node, data in factor_graph.nodes(data=True):
        local_dimension = 1 if data.get("kind") == "factor" else 2
        incident_dimensions = [ranks[frozenset((node, neighbor))] for neighbor in factor_graph.neighbors(node)]
        parameter_count += local_dimension * int(np.prod(incident_dimensions, dtype=int))
    return list(ranks.values()), entropies, parameter_count


def _quimb_compression(state, nodes, order, max_bond):
    ordered = _ordered_state(state, nodes, order)
    mps = qtn.MatrixProductState.from_dense(
        ordered,
        dims=[2] * len(nodes),
        cutoff=1e-12,
    )
    exact_max_bond = int(mps.max_bond())
    compressed = mps.copy()
    compressed.compress(max_bond=max_bond, cutoff=0.0)
    approximation = np.asarray(compressed.to_dense()).reshape(-1)
    approximation /= np.linalg.norm(approximation)
    fidelity = float(abs(np.vdot(ordered, approximation)) ** 2)
    relative_error = float(np.linalg.norm(ordered - approximation) / np.linalg.norm(ordered))
    return exact_max_bond, fidelity, relative_error


def run_state_bond_experiments(max_chain_length=4):
    metric_rows = []
    compression_rows = []
    for chain_length in range(1, max_chain_length + 1):
        for architecture in ("direct", "passive_buffer", "floquet_buffer"):
            primal, factor = build_graphs(chain_length, architecture)
            optimized_order = _optimized_order(primal)
            interleaved_order = _interleaved_order(chain_length, architecture)
            nodes, sampled_states = evolve_state(chain_length, architecture)
            for current_time, state in sampled_states:
                for method, order in (
                    ("interleaved_mps", interleaved_order),
                    ("graph_optimized_mps", optimized_order),
                ):
                    ranks, entropies, parameters = _mps_bond_profile(state, nodes, order)
                    metric_rows.append(
                        {
                            "architecture": architecture,
                            "chain_length": chain_length,
                            "time": current_time,
                            "method": method,
                            "max_required_bond_dimension": max(ranks, default=1),
                            "max_bond_entropy_bits": max(entropies, default=0.0),
                            "raw_tensor_parameter_count": parameters,
                        }
                    )
                ranks, entropies, parameters = _factor_tree_profile(state, nodes, factor)
                metric_rows.append(
                    {
                        "architecture": architecture,
                        "chain_length": chain_length,
                        "time": current_time,
                        "method": "factor_tree_ttn",
                        "max_required_bond_dimension": max(ranks, default=1),
                        "max_bond_entropy_bits": max(entropies, default=0.0),
                        "raw_tensor_parameter_count": parameters,
                    }
                )

            final_state = sampled_states[-1][1]
            for method, order in (
                ("interleaved_mps", interleaved_order),
                ("graph_optimized_mps", optimized_order),
            ):
                for max_bond in (2, 4, 8):
                    exact_bond, fidelity, relative_error = _quimb_compression(
                        final_state, nodes, order, max_bond
                    )
                    compression_rows.append(
                        {
                            "architecture": architecture,
                            "chain_length": chain_length,
                            "ordering": method,
                            "exact_quimb_max_bond": exact_bond,
                            "compression_max_bond": max_bond,
                            "fidelity": fidelity,
                            "relative_l2_error": relative_error,
                        }
                    )
    return metric_rows, compression_rows


def _factor_positions(chain_length, architecture):
    positions = {"U3": (0.0, 0.0)}
    directions = {
        "0": np.array([-0.85, 0.55]),
        "1": np.array([-0.10, -1.00]),
        "z": np.array([0.90, 0.45]),
    }
    for branch, machine in zip(BRANCHES, MACHINE_NODES):
        direction = directions[branch]
        positions[machine] = tuple(direction)
        offset = 1
        if architecture != "direct" and branch == "z":
            positions["F"] = tuple(1.65 * direction)
            offset = 2
        for index in range(1, chain_length + 1):
            positions[_bath_node(branch, index)] = tuple((offset + 0.75 * index) * direction)
    return positions


def make_topology_figure(output_path):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.5, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.25, 0.75))
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    ]
    colors = {"factor": "#9467bd", "machine": "#f2cf5b", "buffer": "#59a14f", "bath": "#9ecae1"}
    for ax, architecture, title in zip(
        axes[:2],
        ("direct", "floquet_buffer"),
        ("Direct factor tree", "Floquet-buffer factor tree"),
    ):
        _, factor = build_graphs(3, architecture)
        positions = _factor_positions(3, architecture)
        node_colors = [colors[factor.nodes[node]["kind"]] for node in factor.nodes]
        nx.draw_networkx(
            factor,
            pos=positions,
            ax=ax,
            node_color=node_colors,
            node_size=670,
            font_size=8,
            edge_color="#555555",
            width=1.4,
        )
        ax.set_title(title)
        ax.set_axis_off()

    primal, _ = build_graphs(3, "floquet_buffer")
    optimized = _optimized_order(primal)
    positions = {node: (index, 0.0) for index, node in enumerate(optimized)}
    node_colors = [colors[primal.nodes[node]["kind"]] for node in optimized]
    nx.draw_networkx_nodes(primal, positions, nodelist=optimized, node_color=node_colors, node_size=510, ax=axes[2])
    nx.draw_networkx_labels(primal, positions, font_size=7, ax=axes[2])
    for left, right in primal.edges:
        x_left, x_right = positions[left][0], positions[right][0]
        radius = 0.18 + 0.035 * abs(x_right - x_left)
        nx.draw_networkx_edges(
            primal,
            positions,
            edgelist=[(left, right)],
            connectionstyle=f"arc3,rad={radius}",
            edge_color="#666666",
            width=1.0,
            arrows=True,
            arrowstyle="-",
            ax=axes[2],
        )
    axes[2].axhline(0.0, color="#bbbbbb", linewidth=0.7, zorder=0)
    axes[2].set_title("Graph-optimized MPS path")
    axes[2].set_ylim(-0.7, 3.2)
    axes[2].set_axis_off()
    fig.savefig(output_path, dpi=210)
    plt.close(fig)
    return output_path


def make_scaling_figure(graph_rows, state_rows, compression_rows, output_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    colors = {"direct": "#4c78a8", "passive_buffer": "#54a24b", "floquet_buffer": "#e45756"}
    styles = {"interleaved_mps": ":", "graph_optimized_mps": "-", "factor_tree_ttn": "--"}

    for architecture in colors:
        for ordering in styles:
            selected = [
                row for row in graph_rows
                if row["architecture"] == architecture and row["ordering"] == ordering
            ]
            axes[0, 0].plot(
                [row["chain_length"] for row in selected],
                [row["cutwidth"] for row in selected],
                color=colors[architecture],
                linestyle=styles[ordering],
                marker="o" if ordering == "graph_optimized_mps" else None,
                label=f"{architecture}, {ordering}" if architecture in {"direct", "floquet_buffer"} else None,
            )
    axes[0, 0].set_xlabel("bath-chain length L")
    axes[0, 0].set_ylabel("interaction cutwidth")
    axes[0, 0].set_title("Path versus tree separators")
    axes[0, 0].legend(fontsize=7, ncol=2)

    for architecture in colors:
        for method in styles:
            grouped = []
            for chain_length in sorted({row["chain_length"] for row in state_rows}):
                values = [
                    row["max_required_bond_dimension"]
                    for row in state_rows
                    if row["architecture"] == architecture
                    and row["chain_length"] == chain_length
                    and row["method"] == method
                ]
                grouped.append(max(values))
            axes[0, 1].plot(
                range(1, len(grouped) + 1),
                grouped,
                color=colors[architecture],
                linestyle=styles[method],
                marker="s" if method == "factor_tree_ttn" else "o",
                label=f"{architecture}, {method}",
            )
    axes[0, 1].set_xlabel("bath-chain length L")
    axes[0, 1].set_ylabel("max Schmidt bond dimension")
    axes[0, 1].set_title("Exact evolved-state requirement")
    axes[0, 1].legend(fontsize=6.5, ncol=3)

    max_length = max(row["chain_length"] for row in state_rows)
    for architecture in colors:
        for method in ("graph_optimized_mps", "factor_tree_ttn"):
            selected = sorted(
                (
                    row for row in state_rows
                    if row["architecture"] == architecture
                    and row["chain_length"] == max_length
                    and row["method"] == method
                ),
                key=lambda row: row["time"],
            )
            axes[1, 0].plot(
                [row["time"] for row in selected],
                [row["raw_tensor_parameter_count"] for row in selected],
                color=colors[architecture],
                linestyle=styles[method],
                label=f"{architecture}, {method}",
            )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("raw tensor entries")
    axes[1, 0].set_title(f"Representation size at L={max_length}")
    axes[1, 0].legend(fontsize=7)

    for architecture in colors:
        selected = [
            row for row in compression_rows
            if row["architecture"] == architecture
            and row["chain_length"] == max_length
            and row["ordering"] == "graph_optimized_mps"
        ]
        axes[1, 1].plot(
            [row["compression_max_bond"] for row in selected],
            [1.0 - row["fidelity"] for row in selected],
            color=colors[architecture],
            marker="o",
            label=architecture,
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Quimb MPS maximum bond")
    axes[1, 1].set_ylabel("one minus fidelity")
    axes[1, 1].set_title("Final-state MPS compression error")
    axes[1, 1].legend(fontsize=8)
    fig.savefig(output_path, dpi=210)
    plt.close(fig)
    return output_path


def _write_report(path, graph_rows, state_rows, compression_rows):
    max_length = max(row["chain_length"] for row in state_rows)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Tensor-network and graph-geometry benchmark\n")
        handle.write("===========================================\n\n")
        handle.write("Graph result:\n")
        handle.write(
            "- Replacing the three-body interaction by one factor tensor makes the complete "
            "machine-plus-chain incidence graph a tree (treewidth 1).\n"
        )
        for architecture in ("direct", "passive_buffer", "floquet_buffer"):
            selected = [
                row for row in graph_rows
                if row["architecture"] == architecture
                and row["chain_length"] == max(row_["chain_length"] for row_ in graph_rows)
            ]
            optimized = next(row for row in selected if row["ordering"] == "graph_optimized_mps")
            interleaved = next(row for row in selected if row["ordering"] == "interleaved_mps")
            handle.write(
                f"- {architecture}: primal-treewidth upper bound={optimized['primal_treewidth_upper_bound']}, "
                f"optimized MPS cutwidth={optimized['cutwidth']}, "
                f"interleaved cutwidth={interleaved['cutwidth']}, TTN separator=1.\n"
            )

        handle.write(f"\nExact sparse-evolution state result at L={max_length}:\n")
        for architecture in ("direct", "passive_buffer", "floquet_buffer"):
            for method in ("interleaved_mps", "graph_optimized_mps", "factor_tree_ttn"):
                selected = [
                    row for row in state_rows
                    if row["architecture"] == architecture
                    and row["chain_length"] == max_length
                    and row["method"] == method
                ]
                handle.write(
                    f"- {architecture}, {method}: max chi="
                    f"{max(row['max_required_bond_dimension'] for row in selected)}, "
                    f"max entropy={max(row['max_bond_entropy_bits'] for row in selected):.6f} bits, "
                    f"max raw entries={max(row['raw_tensor_parameter_count'] for row in selected)}.\n"
                )

        handle.write("\nQuimb MPS compression at the final time:\n")
        for architecture in ("direct", "passive_buffer", "floquet_buffer"):
            selected = [
                row for row in compression_rows
                if row["architecture"] == architecture
                and row["chain_length"] == max_length
                and row["ordering"] == "graph_optimized_mps"
            ]
            for row in selected:
                handle.write(
                    f"- {architecture}, chi_cap={row['compression_max_bond']}: "
                    f"fidelity={row['fidelity']:.10f}, relative_error={row['relative_l2_error']:.3e}.\n"
                )

        handle.write("\nInterpretation:\n")
        handle.write(
            "The Floquet term is local on the buffer, so it does not change spatial graph "
            "treewidth. It can nevertheless increase the numerical Schmidt ranks by generating "
            "more entanglement. A topology-matched TTN isolates each reservoir branch with a "
            "single graph edge, whereas an MPS must choose a linear ordering and can carry "
            "several branch correlations through one bond. The calculation is an exact small-"
            "network evolution followed by exact Schmidt analysis and Quimb MPS factorization; "
            "it is not yet a large-chain TDVP/TTN time evolution.\n"
        )
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    graph_rows = run_graph_scaling()
    state_rows, compression_rows = run_state_bond_experiments()
    graph_path = _write_csv(OUTPUT_DIR / "graph_layout_metrics.csv", graph_rows)
    state_path = _write_csv(OUTPUT_DIR / "exact_state_tn_metrics.csv", state_rows)
    compression_path = _write_csv(OUTPUT_DIR / "mps_compression_metrics.csv", compression_rows)
    topology_path = make_topology_figure(OUTPUT_DIR / "tensor_network_graph_topologies.png")
    scaling_path = make_scaling_figure(
        graph_rows,
        state_rows,
        compression_rows,
        OUTPUT_DIR / "mps_ttn_graph_scaling.png",
    )
    report_path = _write_report(
        OUTPUT_DIR / "tensor_network_graph_report.txt",
        graph_rows,
        state_rows,
        compression_rows,
    )
    print(report_path.read_text(encoding="utf-8"), end="")
    print(f"Saved graph metrics: {graph_path}")
    print(f"Saved state metrics: {state_path}")
    print(f"Saved MPS compression metrics: {compression_path}")
    print(f"Saved topology figure: {topology_path}")
    print(f"Saved scaling figure: {scaling_path}")


if __name__ == "__main__":
    main()
