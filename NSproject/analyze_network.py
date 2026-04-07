"""
MyAnimeList 2020 — Network Science Exploratory Analysis
Constructs an anime co-watch network and computes graph-theoretic properties.
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/Users/shamiksinha/Desktop/NSporject/Anime Recommendation Database 2020/"

# ─────────────────────────────────────────────
# 1. INSPECT & BASIC STATS
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: DATASET INSPECTION")
print("=" * 60)

# --- anime.csv ---
anime = pd.read_csv(DATA_DIR + "anime.csv", low_memory=False)
print(f"\nanime.csv: {anime.shape[0]:,} rows × {anime.shape[1]} columns")
print("Columns:", list(anime.columns))
print("\nData types:")
print(anime.dtypes)
print("\nMissing values (top 10):")
print(anime.isnull().sum().sort_values(ascending=False).head(10))
print("\nNumeric summary (key cols):")
key_cols = ["Score", "Members", "Favorites", "Episodes", "Watching", "Completed"]
key_cols = [c for c in key_cols if c in anime.columns]
print(anime[key_cols].describe())

# Type distribution
if "Type" in anime.columns:
    print("\nAnime types:")
    print(anime["Type"].value_counts())

# Genre analysis
if "Genres" in anime.columns:
    all_genres = []
    for g in anime["Genres"].dropna():
        all_genres.extend([x.strip() for x in g.split(",")])
    genre_counts = Counter(all_genres)
    print(f"\nUnique genres: {len(genre_counts)}")
    print("Top 15 genres:", dict(sorted(genre_counts.items(), key=lambda x: -x[1])[:15]))

# --- rating_complete.csv (header + sample) ---
print("\n" + "=" * 60)
print("rating_complete.csv stats:")
rc_header = pd.read_csv(DATA_DIR + "rating_complete.csv", nrows=0)
print("Columns:", list(rc_header.columns))

# Count unique users efficiently by sampling
print("Sampling 500k rows to estimate user/anime diversity...")
rc_sample_big = pd.read_csv(DATA_DIR + "rating_complete.csv", nrows=500_000)
print(f"  Unique users in first 500k rows: {rc_sample_big['user_id'].nunique():,}")
print(f"  Unique anime in first 500k rows: {rc_sample_big['anime_id'].nunique():,}")
print(f"  Rating distribution:\n{rc_sample_big['rating'].value_counts().sort_index()}")

# ─────────────────────────────────────────────
# 2. CO-WATCH NETWORK CONSTRUCTION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: CO-WATCH NETWORK CONSTRUCTION")
print("=" * 60)

# Strategy: take the top-N most active users from rating_complete.csv
# (users with most ratings), build their watch lists, then project onto anime.
# We use rating_complete because it only contains users who both completed
# AND scored an anime — giving us cleaner "genuine watch" signal.

SAMPLE_USERS = 50_000   # number of users to include
MIN_COWATCH   = 15      # minimum shared viewers to create an edge
MIN_USER_RATINGS = 10   # discard users with very few ratings (noise)
MIN_ANIME_WATCHERS = 50 # discard very obscure anime

print(f"\nParameters:")
print(f"  Sample users     : {SAMPLE_USERS:,}")
print(f"  Min co-watches   : {MIN_COWATCH}")
print(f"  Min ratings/user : {MIN_USER_RATINGS}")
print(f"  Min watchers/anime: {MIN_ANIME_WATCHERS}")

# Load full rating_complete in chunks, collect user→anime mapping
print("\nLoading rating_complete.csv in chunks (this may take a minute)...")

CHUNK = 1_000_000
user_anime = defaultdict(set)  # user_id → set of anime_ids
anime_users = defaultdict(set) # anime_id → set of user_ids
rows_read = 0

for chunk in pd.read_csv(DATA_DIR + "rating_complete.csv", chunksize=CHUNK):
    for _, row in chunk.iterrows():
        u, a = int(row["user_id"]), int(row["anime_id"])
        user_anime[u].add(a)
        anime_users[a].add(u)
    rows_read += len(chunk)
    print(f"  Read {rows_read:,} rows, {len(user_anime):,} users seen so far...")
    if rows_read >= 10_000_000:  # cap at 10M rows for feasibility
        break

print(f"\nTotal rows read: {rows_read:,}")
print(f"Total unique users: {len(user_anime):,}")
print(f"Total unique anime: {len(anime_users):,}")

# ── Filter by activity ──
user_counts = {u: len(animes) for u, animes in user_anime.items()}
active_users = [u for u, cnt in user_counts.items() if cnt >= MIN_USER_RATINGS]
print(f"\nUsers with >= {MIN_USER_RATINGS} ratings: {len(active_users):,}")

# Sort by activity, take top SAMPLE_USERS
active_users_sorted = sorted(active_users, key=lambda u: user_counts[u], reverse=True)
selected_users = set(active_users_sorted[:SAMPLE_USERS])
print(f"Selected top {len(selected_users):,} most active users")

# Distribution of selected user activity
sel_counts = [user_counts[u] for u in selected_users]
print(f"  Ratings/user → min={min(sel_counts)}, median={np.median(sel_counts):.0f}, "
      f"mean={np.mean(sel_counts):.1f}, max={max(sel_counts)}")

# ── Rebuild anime→users for selected users only ──
print("\nRebuilding anime→user sets for selected users...")
anime_sel_users = defaultdict(set)
for u in selected_users:
    for a in user_anime[u]:
        anime_sel_users[a].add(u)

# Filter out niche anime
popular_anime = {a: users for a, users in anime_sel_users.items()
                 if len(users) >= MIN_ANIME_WATCHERS}
print(f"Anime with >= {MIN_ANIME_WATCHERS} watchers (in selected users): {len(popular_anime):,}")

anime_list = sorted(popular_anime.keys())
anime_idx  = {a: i for i, a in enumerate(anime_list)}
print(f"Anime nodes that will appear in the network: {len(anime_list):,}")

# ── Compute co-watch counts via set intersections ──
print(f"\nComputing pairwise co-watch counts (min threshold = {MIN_COWATCH})...")
print("This is O(|anime|² · avg_users_per_anime) — may take several minutes...")

edges = []
n_anime = len(anime_list)

# Use inverted index approach: for each user, generate all pairs of anime they watched
# This is faster than pairwise set intersections for sparse data
pair_counts = Counter()
processed = 0
for u in selected_users:
    watched = [a for a in user_anime[u] if a in popular_anime]
    watched.sort()
    for i in range(len(watched)):
        for j in range(i + 1, len(watched)):
            pair_counts[(watched[i], watched[j])] += 1
    processed += 1
    if processed % 10_000 == 0:
        print(f"  Processed {processed:,}/{len(selected_users):,} users, "
              f"{len(pair_counts):,} candidate pairs so far...")

print(f"\nTotal candidate pairs: {len(pair_counts):,}")
significant_pairs = [(a, b, w) for (a, b), w in pair_counts.items() if w >= MIN_COWATCH]
print(f"Pairs with >= {MIN_COWATCH} co-watchers: {len(significant_pairs):,}")

# ─────────────────────────────────────────────
# 3. BUILD NETWORKX GRAPH & COMPUTE PROPERTIES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: GRAPH CONSTRUCTION & BASIC PROPERTIES")
print("=" * 60)

G = nx.Graph()
G.add_nodes_from(anime_list)
for a, b, w in significant_pairs:
    G.add_edge(a, b, weight=w)

# Remove isolates for most analyses
G_connected = G.copy()
isolates = list(nx.isolates(G_connected))
G_connected.remove_nodes_from(isolates)

print(f"\nFull graph:      {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
print(f"After removing {len(isolates):,} isolates:")
print(f"Connected graph: {G_connected.number_of_nodes():,} nodes, {G_connected.number_of_edges():,} edges")
print(f"Density: {nx.density(G_connected):.6f}")

# Degree stats
degrees = [d for _, d in G_connected.degree()]
print(f"\nDegree statistics:")
print(f"  min={min(degrees)}, max={max(degrees)}, mean={np.mean(degrees):.2f}, "
      f"median={np.median(degrees):.0f}, std={np.std(degrees):.2f}")

# ── Connected components ──
components = sorted(nx.connected_components(G_connected), key=len, reverse=True)
print(f"\nConnected components: {len(components)}")
print(f"  Largest component : {len(components[0]):,} nodes "
      f"({100*len(components[0])/G_connected.number_of_nodes():.1f}% of connected nodes)")
if len(components) > 1:
    print(f"  2nd largest       : {len(components[1]):,} nodes")
    print(f"  Component sizes   : {[len(c) for c in components[:10]]}")

# Work on the giant component for the rest
GC = G_connected.subgraph(components[0]).copy()
print(f"\nGiant component: {GC.number_of_nodes():,} nodes, {GC.number_of_edges():,} edges")

# ── Degree distribution — check for power law ──
gc_degrees = sorted([d for _, d in GC.degree()], reverse=True)
print(f"\nDegree distribution (giant component):")
print(f"  Top 10 degree values: {gc_degrees[:10]}")
percentiles = [50, 75, 90, 95, 99]
for p in percentiles:
    print(f"  {p}th percentile: {np.percentile(gc_degrees, p):.0f}")

# Log-bin degree distribution to check for power-law
degree_counts = Counter(gc_degrees)
deg_vals = np.array(sorted(degree_counts.keys()))
deg_freq = np.array([degree_counts[d] for d in deg_vals])

# Simple power-law fit: log-log regression on tail
# Use degrees > mean to focus on tail
mean_deg = np.mean(gc_degrees)
tail_mask = deg_vals >= mean_deg
if tail_mask.sum() >= 5:
    log_x = np.log(deg_vals[tail_mask])
    log_y = np.log(deg_freq[tail_mask])
    # linear fit in log-log space
    slope, intercept = np.polyfit(log_x, log_y, 1)
    print(f"\nPower-law tail fit (log-log regression, degrees >= mean={mean_deg:.1f}):")
    print(f"  Exponent γ ≈ {-slope:.3f}  (scale-free networks typically 2 < γ < 3)")
    print(f"  R² = {np.corrcoef(log_x, log_y)[0,1]**2:.4f}")

# ── Clustering coefficient ──
print("\nComputing clustering coefficients...")
avg_clustering = nx.average_clustering(GC)
print(f"  Average clustering coefficient: {avg_clustering:.4f}")

# Compare to random Erdos-Renyi baseline
n, m = GC.number_of_nodes(), GC.number_of_edges()
p_rand = 2 * m / (n * (n - 1))
c_rand = p_rand  # E-R clustering ≈ p
print(f"  Expected C for random graph (p={p_rand:.6f}): {c_rand:.6f}")
print(f"  Ratio C_actual / C_random: {avg_clustering / c_rand:.1f}x")

# ── Transitivity (global clustering) ──
transitivity = nx.transitivity(GC)
print(f"  Global transitivity (fraction of closed triangles): {transitivity:.4f}")

# ── Top hubs by degree ──
print("\nTop 20 hubs by degree (with anime names):")
anime_names = dict(zip(anime["MAL_ID"], anime["Name"]))
gc_deg_sorted = sorted(GC.degree(), key=lambda x: -x[1])
for mal_id, deg in gc_deg_sorted[:20]:
    name = anime_names.get(mal_id, f"ID={mal_id}")
    score = anime.loc[anime["MAL_ID"] == mal_id, "Score"].values
    score_str = f"{score[0]}" if len(score) > 0 else "N/A"
    print(f"  {name[:45]:45s}  degree={deg:4d}  score={score_str}")

# ── Top hubs by weighted degree (strength) ──
print("\nTop 10 hubs by weighted degree (strength = total co-watches):")
strength = {n: sum(d["weight"] for _, _, d in GC.edges(n, data=True)) for n in GC.nodes()}
for mal_id, s in sorted(strength.items(), key=lambda x: -x[1])[:10]:
    name = anime_names.get(mal_id, f"ID={mal_id}")
    print(f"  {name[:45]:45s}  strength={s:,}")

# ── Assortativity ──
assort = nx.degree_assortativity_coefficient(GC)
print(f"\nDegree assortativity: {assort:.4f}")
print("  (> 0 = assortative [hubs connect to hubs], < 0 = disassortative)")

# ── Community structure (Louvain via greedy modularity) ──
print("\nDetecting communities (greedy modularity maximization)...")
from networkx.algorithms.community import greedy_modularity_communities
communities = list(greedy_modularity_communities(GC))
modularity = nx.community.modularity(GC, communities)
print(f"  Number of communities: {len(communities)}")
print(f"  Modularity Q: {modularity:.4f}  (Q > 0.3 = strong structure)")
comm_sizes = sorted([len(c) for c in communities], reverse=True)
print(f"  Community sizes (top 10): {comm_sizes[:10]}")

# What are the top communities about?
print("\nTop 3 community profiles (top 5 members by degree within community):")
for i, comm in enumerate(communities[:3]):
    sub = GC.subgraph(comm)
    top_nodes = sorted(sub.degree(), key=lambda x: -x[1])[:5]
    names = [anime_names.get(n, str(n)) for n, _ in top_nodes]
    genres_in_comm = []
    for n, _ in top_nodes:
        g = anime.loc[anime["MAL_ID"] == n, "Genres"].values
        if len(g) > 0 and pd.notna(g[0]):
            genres_in_comm.extend([x.strip() for x in g[0].split(",")])
    top_genres = Counter(genres_in_comm).most_common(3)
    print(f"  Community {i+1} ({len(comm)} nodes): {names}")
    print(f"    Dominant genres: {top_genres}")

# ── Small-world check: average shortest path on sample ──
print("\nSmall-world check (sample of 500 nodes for path length)...")
sample_nodes = list(GC.nodes())[:500]
GC_sub = GC.subgraph(sample_nodes)
if nx.is_connected(GC_sub):
    avg_path = nx.average_shortest_path_length(GC_sub)
else:
    # Use largest component of subgraph
    sc = max(nx.connected_components(GC_sub), key=len)
    GC_sub2 = GC_sub.subgraph(sc)
    avg_path = nx.average_shortest_path_length(GC_sub2)
    print(f"  (subgraph not connected; used component of {len(sc)} nodes)")
print(f"  Average shortest path (500-node sample): {avg_path:.3f}")
log_n = np.log(GC.number_of_nodes())
print(f"  ln(N) = {log_n:.2f}  — small world if avg_path ≈ ln(N)/ln(mean_degree)")

# ─────────────────────────────────────────────
# 4. SUMMARY & RECOMMENDATIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: NETWORK SCIENCE SUMMARY")
print("=" * 60)

print(f"""
DATASET SUMMARY:
  anime.csv         : {anime.shape[0]:,} anime titles
  rating_complete   : 57.6M rows (users who completed + scored anime)
  animelist         : 109M rows (all users, all statuses)

SAMPLING STRATEGY USED:
  Rows read from rating_complete : {rows_read:,}
  Active users selected          : {len(selected_users):,}
  Popular anime included         : {len(popular_anime):,}

NETWORK CONSTRUCTED (co-watch, threshold={MIN_COWATCH}):
  Nodes (anime)    : {GC.number_of_nodes():,}
  Edges (co-watch) : {GC.number_of_edges():,}
  Density          : {nx.density(GC):.6f}
  Avg degree       : {np.mean([d for _,d in GC.degree()]):.2f}
  Max degree       : {max(d for _,d in GC.degree())}
  Avg clustering   : {avg_clustering:.4f}
  Transitivity     : {transitivity:.4f}
  Assortativity    : {assort:.4f}
  Communities      : {len(communities)}  (Q={modularity:.4f})
""")

print("FEASIBILITY FOR FULL ANALYSIS:")
print(f"  rating_complete has 57.6M rows across ~320k users.")
print(f"  With {SAMPLE_USERS:,} users, co-watch pairs = {len(pair_counts):,} candidate pairs.")
print(f"  Full dataset (~320k users) would produce ~{len(pair_counts)*320//SAMPLE_USERS:,} candidate pairs.")
print(f"  At 16 bytes/pair, that's ~{len(pair_counts)*320//SAMPLE_USERS*16/1e9:.2f} GB RAM — borderline.")
print(f"  RECOMMENDATION: 50k–100k users is the sweet spot for a laptop.")
print(f"  For 100k users use chunked processing or sparse matrix approach (scipy.sparse).")
