import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances

# Page configuration
st.set_page_config(page_title="Football Player Analytics Dashboard", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #555;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .welcome-box {
        padding: 30px;
        background-color: #333;
        border-radius: 15px;
        border-left: 5px solid #1976d2;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Description
st.title("⚽ Football Player Analytics Dashboard (2024-2025)")

file_path = "https://raw.githubusercontent.com/kyaw-pyae-sone/football-analytics-app/refs/heads/main/dataframe.csv"

# Data Loading function
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Basic data cleaning
    df = df.dropna(subset=['Age', 'MP', 'Min'])
    # Extract League Name from 'Comp' column if exists
    if 'Comp' in df.columns:
        df['League'] = df['Comp'].str.replace(r'^[a-z]{2}\s', '', regex=True)
    return df

# Helper for processing clustering with expanded features
def process_clustering(df, n_clusters=4):
    # Features: Gls, Ast, xG, xAG, PrgC, PrgP, Tkl, Int, Clr, Touches
    features = ['Gls', 'Ast', 'xG', 'xAG', 'PrgC', 'PrgP', 'Tkl', 'Int', 'Clr', 'Touches']
    
    # Ensure we only use numeric features that exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    
    # Fill missing values with 0
    X = df[available_features].fillna(0)
    
    # Scaling for ML
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # PCA for visualization (2D)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    
    return df, available_features, scaler

# Sidebar Section
st.sidebar.title("Settings & Upload")
uploaded_file = st.sidebar.file_uploader('1. Please upload the CSV dataset, type="csv"')

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    st.sidebar.success("Manual CSV Loaded!")
else:
    # Auto-load from GitHub path
    try:
        df_raw = load_data(file_path)
        st.sidebar.info("Default Dataset Loaded (Auto)")
    except Exception as e:
        st.error(f"Error loading default data: {e}")
        df_raw = None
    
if df_raw is not None:
    # Sidebar Filters
    st.sidebar.header("2. Select the filters.")
    all_leagues = sorted(df_raw['League'].unique()) if 'League' in df_raw.columns else []
    selected_league = st.sidebar.multiselect("Choose a league", options=all_leagues)
    
    selected_squad = st.sidebar.multiselect("Choose a team", options=sorted(df_raw['Squad'].unique()))
    selected_pos = st.sidebar.multiselect("Pick a position", options=sorted(df_raw['Pos'].unique()))
    
    # Filter data logic
    df_filtered = df_raw.copy()
    if selected_league:
        df_filtered = df_filtered[df_filtered['League'].isin(selected_league)]
    if selected_squad:
        df_filtered = df_filtered[df_filtered['Squad'].isin(selected_squad)]
    if selected_pos:
        df_filtered = df_filtered[df_filtered['Pos'].isin(selected_pos)]

    if df_filtered.empty:
        st.warning("⚠️ No data available for the selected filters. Please adjust your filter settings.")
    else:
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Overview", 
            "🤖 ML Clustering", 
            "🔍 Player Search", 
            "🤝 Similar Players",
            "🏢 Squad Analytics",
            "Best Players wise Position per League",
            "Position-based Role Discovery (GK,MF,DF)",
            "Goal Keeper Performance"

        ])

        with tab1:
            st.subheader("Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Players", len(df_filtered))
            col2.metric("Avg Age", round(df_filtered['Age'].mean(), 1))
            col3.metric("Max Goals", df_filtered['Gls'].max())
            col4.metric("Avg Minutes", int(df_filtered['Min'].mean()))

            st.divider()

            st.write("Statistics")
        
            full_stat_summary = df_raw.select_dtypes(include=[np.number])
            
            
            if 'Rk' in full_stat_summary.columns:
                full_stat_summary = full_stat_summary.drop(columns=['Rk'])
                
            
            st.dataframe(full_stat_summary.describe().T.style.format("{:.2f}"), use_container_width=True)

            st.divider()
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.write("**Top Scorer များ (Goals)**")
                top_scorers = df_filtered.nlargest(10, 'Gls')[['Player', 'Squad', 'Gls']]
                fig_goals = px.bar(top_scorers, x='Gls', y='Player', color='Squad', orientation='h', height=400)
                st.plotly_chart(fig_goals, use_container_width=True)
                
            with col_right:
                st.write("**Age vs xG (Expected Goals Analysis)**")
                y_axis_scatter = 'xG' if 'xG' in df_filtered.columns else 'Gls'
                fig_scatter = px.scatter(df_filtered, x="Age", y=y_axis_scatter, color="Pos", 
                                        hover_data=['Player', 'Squad'],
                                        size='Min', opacity=0.7)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.divider()
            st.subheader("📍 Player Position Analysis")
            
            st.write("**Total Positions Distribution**")
            fig_pos_all, ax_pos_all = plt.subplots(figsize=(12, 5))
            df_filtered["Pos"].value_counts().plot(kind='bar', ax=ax_pos_all, color='#3498db', edgecolor='black')
            ax_pos_all.set_title("Distribution of All Recorded Positions", fontsize=14, pad=15)
            ax_pos_all.set_xlabel("Positions", fontsize=12)
            ax_pos_all.set_ylabel("Number of Players", fontsize=12)
            plt.xticks(rotation=45)
            st.pyplot(fig_pos_all)

            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.write("**Hybrid Positions Only**")
                fig_pos_hyb, ax_pos_hyb = plt.subplots(figsize=(10, 6))
                df_filtered[df_filtered["Pos"].str.contains(",")]["Pos"].value_counts().plot(kind='bar', ax=ax_pos_hyb, color='#e67e22', edgecolor='black')
                ax_pos_hyb.set_title("Hybrid Position Distribution (e.g., FW,MF)", fontsize=13)
                ax_pos_hyb.set_xlabel("Player Positions")
                ax_pos_hyb.set_ylabel("Count")
                st.pyplot(fig_pos_hyb)

            with col_c2:
                st.write("**Single Positions Only**")
                fig_pos_single, ax_pos_single = plt.subplots(figsize=(10, 6))
                df_filtered[~df_filtered["Pos"].str.contains(",")]["Pos"].value_counts().plot(kind='bar', ax=ax_pos_single, color='#2ecc71', edgecolor='black')
                ax_pos_single.set_title("Single Position Distribution (Primary Role)", fontsize=13)
                ax_pos_single.set_xlabel("Player Positions")
                ax_pos_single.set_ylabel("Count")
                st.pyplot(fig_pos_single)

            st.divider()
            st.subheader("🌍 League & Squad Distribution")
            
            st.write("Number of Squads per League**")
            if 'League' in df_filtered.columns and 'Squad' in df_filtered.columns:
                league_squad_counts = df_filtered.groupby('League')['Squad'].nunique().sort_values(ascending=False).reset_index()
                league_squad_counts.columns = ['League', 'Squad Count']
                
                fig_league_dist = px.bar(
                    league_squad_counts, 
                    x='League', 
                    y='Squad Count',
                    color='Squad Count',
                    color_continuous_scale='Viridis',
                    text='Squad Count',
                    labels={'Squad Count': 'Team Count', 'League': 'League Name'}
                )
                fig_league_dist.update_traces(textposition='outside')
                fig_league_dist.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_league_dist, use_container_width=True)

        with tab2:
            st.subheader("🤖 K-Means Clustering Analysis (Min >= 450)")
            st.info("Features: Gls, Ast, xG, xAG, PrgC, PrgP, Tkl, Int, Clr, Touches")
            
            # Using your specified K=3 logic
            n_clusters = st.slider("Select the number of clusters", 2, 8, 3)
            
            df_clustered, cluster_features, scaler = process_clustering(df_filtered, n_clusters)
            
            st.write(f"Number of players after filtering (Min >= 450): {len(df_clustered)}")
            
            # PCA Visualization
            st.write("**၁။ Player Clusters (PCA Projection)**")
            fig_pca = px.scatter(df_clustered, x='PCA1', y='PCA2', color='Cluster',
                                    hover_data=['Player', 'Squad', 'Pos', 'Gls', 'Ast'],
                                    template="plotly_white",
                                    height=500,
                                    color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig_pca, use_container_width=True)
            
            st.divider()
            
            # Radar and Bar Charts for Cluster Characteristics
            col_chart1, col_chart2 = st.columns(2)
            
            # Characteristic table calculations
            cluster_means = df_clustered.groupby('Cluster')[cluster_features].mean()
            # Normalization (0 to 1 scale based on max mean across clusters)
            norm_means = cluster_means / cluster_means.max()

            with col_chart1:
                st.write("**၂။ Cluster Characteristics (Radar Chart)**")
                fig_radar_clusters = go.Figure()
                
                # Custom titles based on your provided logic for K=3
                default_titles = [f'Cluster {i}' for i in range(n_clusters)]
                if n_clusters == 3:
                    default_titles = [
                        'Cluster 0: Low-Volume/Support', 
                        'Cluster 1: Defensive & Buildup Anchors', 
                        'Cluster 2: High-Impact Attackers'
                    ]
                
                for i, cluster_idx in enumerate(norm_means.index):
                    r_values = norm_means.loc[cluster_idx].values.tolist()
                    r_values += [r_values[0]]
                    theta_values = cluster_features + [cluster_features[0]]
                    
                    fig_radar_clusters.add_trace(go.Scatterpolar(
                        r=r_values,
                        theta=theta_values,
                        fill='toself',
                        name=default_titles[i] if i < len(default_titles) else f'Cluster {cluster_idx}'
                    ))
                
                fig_radar_clusters.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
                    showlegend=True
                )
                st.plotly_chart(fig_radar_clusters, use_container_width=True)
            
            with col_chart2:
                st.write("**3. Cluster Characteristics (Bar Chart)**")
                bar_data = norm_means.reset_index().melt(id_vars='Cluster', var_name='Metric', value_name='Normalized Value')
                bar_data['Cluster'] = bar_data['Cluster'].astype(str)
                
                fig_bar_clusters = px.bar(
                    bar_data, 
                    x='Metric', 
                    y='Normalized Value', 
                    color='Cluster',
                    barmode='group',
                    template="plotly_white"
                )
                st.plotly_chart(fig_bar_clusters, use_container_width=True)
            
            with st.expander("Average characteristics of each cluster (Detailed Table)"):
                st.dataframe(cluster_means.style.background_gradient(cmap='Blues'))

            st.write("**Sample Players from each Cluster:**")
            for i in range(n_clusters):
                with st.expander(f"Cluster {i} Samples"):
                    st.table(df_clustered[df_clustered['Cluster'] == i][['Player', 'Pos', 'Squad']].head(10))

        with tab3:
            st.subheader("🔍 Find a player")
            player_name = st.selectbox("Choose a player", options=sorted(df_filtered['Player'].unique()), key="search_select")
            
            if player_name:
                p_data = df_filtered[df_filtered['Player'] == player_name].iloc[0]
                
                p_col1, p_col2 = st.columns([1, 2])
                with p_col1:
                    st.success(f"### {p_data['Player']}")
                    st.write(f"**Squad:** {p_data['Squad']}")
                    st.write(f"**Position:** {p_data['Pos']}")
                    st.write(f"**League:** {p_data['League']}")
                    st.write(f"**Age:** {p_data['Age']}")
                    st.write(f"**Matches Played:** {p_data['MP']}")
                
                with p_col2:
                    radar_features_p = ['Gls', 'xG', 'Ast', 'xAG', 'PrgC', 'PrgP', 'Tkl', 'Int', 'Clr']
                    available_radar = [f for f in radar_features_p if f in df_filtered.columns]
                    p_values = p_data[available_radar].values
                    fig_radar = px.line_polar(r=p_values, theta=available_radar, line_close=True)
                    fig_radar.update_traces(fill='toself')
                    st.plotly_chart(fig_radar, use_container_width=True)

        with tab4:
            st.subheader("🤝 Find players with similar attributes")
            target_player = st.selectbox("Select a reference player", options=sorted(df_filtered['Player'].unique()), key="sim_select")
            num_sim = st.number_input("Number of similar players to display:", 1, 10, 5)
            
            if target_player:
                features_sim = ['Gls', 'Ast', 'xG', 'xAG', 'PrgC', 'PrgP', 'Tkl', 'Int', 'Clr', 'Touches']
                available_sim = [f for f in features_sim if f in df_filtered.columns]
                X_sim = df_filtered[available_sim].fillna(0)
                
                scaler_sim = StandardScaler()
                X_scaled_sim = scaler_sim.fit_transform(X_sim)
                
                target_idx = df_filtered[df_filtered['Player'] == target_player].index[0]
                target_pos = df_filtered.index.get_loc(target_idx)
                target_vec = X_scaled_sim[target_pos].reshape(1, -1)
                
                distances = euclidean_distances(target_vec, X_scaled_sim).flatten()
                indices = distances.argsort()[1:num_sim+1]
                
                similar_df = df_filtered.iloc[indices].copy()
                similar_df['Similarity Score'] = np.round(1 / (1 + distances[indices]), 3)
                
                st.write(f" Most similar players to: **{target_player}**")
                st.dataframe(similar_df[['Player', 'Squad', 'Pos', 'Gls', 'xG', 'Ast', 'Similarity Score']], use_container_width=True)
                
                st.write("Performance Analysis (Key Stats)")
                compare_df = pd.concat([df_filtered[df_filtered['Player'] == target_player], similar_df])
                fig_compare = px.bar(compare_df, x='Player', y=available_sim[:5], barmode='group')
                st.plotly_chart(fig_compare, use_container_width=True)

        with tab5:
            st.subheader("🏢 Squad Analysis")
            if 'Squad' in df_filtered.columns:
                squad_stats = df_filtered.groupby('Squad').agg({
                    'Player': 'count',
                    'Age': 'mean',
                    'Gls': 'sum',
                    'xG': 'sum',
                    'Ast': 'sum'
                }).rename(columns={'Player': 'Squad Size', 'Age': 'Avg Age', 'Gls': 'Total Goals', 'xG': 'Total xG', 'Ast': 'Total Assists'})
                
                st.write("Squad Performance Overview")
                st.dataframe(squad_stats.sort_values(by='Total Goals', ascending=False), use_container_width=True)
                
                st.write("Team Goals vs. xG (Goals vs xG)")
                fig_squad = px.scatter(squad_stats, x="Total xG", y="Total Goals", 
                                      size="Squad Size", hover_name=squad_stats.index,
                                      text=squad_stats.index)
                fig_squad.update_traces(textposition='top center')
                st.plotly_chart(fig_squad, use_container_width=True)
        
        with tab6:
            st.subheader("🏆 Position-wise Performance Score Analysis")
            st.info("Performance scores are calculated based on position-specific key stats (per 90 minutes) for FW, MF, DF, and GK.")
            
            # 1. Filter Min >= 450
            df_best = df_filtered[df_filtered['Min'] >= 450].copy()
            
            
            metrics_to_90 = ['Gls', 'Ast', 'xG', 'xAG', 'PrgP', 'PrgC', 'Tkl', 'Int', 'Clr', 'SCA', 'SoT']
            if '90s' in df_best.columns:
                for m in metrics_to_90:
                    if m in df_best.columns:
                        df_best[f'{m}/90'] = df_best[m] / df_best['90s']
            
            # 3. Custom Weights Logic
            def calculate_advanced_score(row):
                pos = str(row['Pos'])
                score = 0
                if 'FW' in pos:
                    score = (row.get('Gls/90', 0) * 0.4) + (row.get('xG/90', 0) * 0.3) + (row.get('SCA/90', 0) * 0.3)
                elif 'MF' in pos:
                    score = (row.get('PrgP/90', 0) * 0.3) + (row.get('Ast/90', 0) * 0.3) + (row.get('xAG/90', 0) * 0.2) + (row.get('SCA/90', 0) * 0.2)
                elif 'DF' in pos:
                    score = (row.get('Tkl/90', 0) * 0.3) + (row.get('Int/90', 0) * 0.3) + (row.get('Clr/90', 0) * 0.2) + (row.get('PrgP/90', 0) * 0.2)
                elif 'GK' in pos:
                    score = (row.get('Clr/90', 0) * 0.5) + (row.get('PrgP/90', 0) * 0.5)
                return score

            df_best['Performance_Score'] = df_best.apply(calculate_advanced_score, axis=1)

            # 4. Find Best Players per League & Position
            if 'Comp' in df_best.columns:
                leagues = sorted(df_best['Comp'].unique())
                target_positions = ['FW', 'MF', 'DF', 'GK']
                
                st.write("**Best Players by Position per League**")
                
                # Create separate charts for each league
                for league in leagues:
                    league_best_list = []
                    for pos in target_positions:
                        top_p = df_best[(df_best['Comp'] == league) & 
                                         (df_best['Pos'].str.contains(pos))].nlargest(1, 'Performance_Score')
                        if not top_p.empty:
                            league_best_list.append(top_p)
                    
                    if league_best_list:
                        league_summary = pd.concat(league_best_list)
                        st.markdown(f"#### 📍 {league}")
                        
                        fig_league = px.bar(
                            league_summary, 
                            x='Performance_Score', 
                            y='Player', 
                            color='Pos', 
                            orientation='h',
                            text_auto='.2f',
                            template="plotly_white",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig_league.update_layout(
                            xaxis_title="Performance Score",
                            yaxis_title="",
                            height=250,
                            margin=dict(l=20, r=20, t=30, b=20)
                        )
                        st.plotly_chart(fig_league, use_container_width=True)
                
                st.divider()
                st.write("**Top Players Summary Table**")
                all_best = []
                for league in leagues:
                    for pos in target_positions:
                        all_best.append(df_best[(df_best['Comp'] == league) & (df_best['Pos'].str.contains(pos))].nlargest(1, 'Performance_Score'))
                
                if all_best:
                    summary_df = pd.concat(all_best)
                    st.dataframe(summary_df[['Player', 'Squad', 'Comp', 'Pos', 'Performance_Score']].sort_values(['Comp', 'Performance_Score'], ascending=[True, False]), use_container_width=True)
            else:
                st.error("The 'Comp' column is missing from the dataset.")

        with tab7:
            st.subheader("🤖 Clustering by 3 Position Groups")
            st.write("Clustering players based on position-specific stats: GK (Saves/Clean Sheets), DF (Tackles/Clearances), and MF (Passing/Progression).")
            
            def cluster_by_position_groups(df):
                temp_df = df.copy()
                
                
                def map_pos_to_group(pos):
                    pos = str(pos).upper()
                    if 'GK' in pos: return 'Goalkeeper (GK)'
                    if 'DF' in pos: return 'Defender (DF)'
                    if 'MF' in pos: return 'Midfielder (MF)'
                    return 'Exclude/Forward'
                
                temp_df['Position_Group'] = temp_df['Pos'].apply(map_pos_to_group)
                temp_df = temp_df[temp_df['Position_Group'] != 'Exclude/Forward']
                

                features = [
                    'Gls', 'Ast', 'PrgP', 'PrgC', 'Tkl', 'Int', 'Clr', 'Touches', 
                    'Cmp', 'Cmp%', 'TotDist', 'Err',
                    'Saves', 'Save%', 'CS', 'PSxG-GA' # GK Specific Stats
                ]
                
                
                available_features = [f for f in features if f in temp_df.columns]
                
                if temp_df.empty:
                    return temp_df, available_features

                
                X = temp_df[available_features].fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # PCA for Visualization
                pca = PCA(n_components=2)
                pca_res = pca.fit_transform(X_scaled)
                temp_df['PCA1'] = pca_res[:, 0]
                temp_df['PCA2'] = pca_res[:, 1]
                
                return temp_df, available_features

            df_grouped, clus_feats = cluster_by_position_groups(df_filtered)
            
            if not df_grouped.empty:
                # PCA Visualization based on Position Groups
                fig_pca = px.scatter(
                    df_grouped, x='PCA1', y='PCA2', 
                    color='Position_Group',
                    hover_data=['Player', 'Squad', 'Pos'] + [f for f in ['Saves', 'Save%', 'CS'] if f in df_grouped.columns],
                    title="Player Distribution including GK Save Stats",
                    color_discrete_map={
                        'Goalkeeper (GK)': '#EF553B',
                        'Defender (DF)': '#636EFA',
                        'Midfielder (MF)': '#00CC96'
                    },
                    height=600
                )
                st.plotly_chart(fig_pca, use_container_width=True)
                
                # Download Data Section
                st.write("---")
                st.write("**Data Export Options**")
                csv_data = df_grouped.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export GK/DF/MF cluster data as CSV",
                    data=csv_data,
                    file_name='football_positional_analysis.csv',
                    mime='text/csv',
                )

                st.divider()
                
                # Stats comparison by group
                st.write("**Average stats for the 3 clusters (including save stats)**")
                group_summary = df_grouped.groupby('Position_Group')[clus_feats].mean()
                st.dataframe(group_summary.style.highlight_max(axis=0), use_container_width=True)
                
                # Filter by group to see players
                selected_group = st.selectbox("View players by cluster", df_grouped['Position_Group'].unique())
                st.write(f"**{selected_group} Cluster Group Members**")
                st.dataframe(df_grouped[df_grouped['Position_Group'] == selected_group][['Player', 'Squad', 'Pos'] + clus_feats], use_container_width=True)
            else:
                st.error("No players found in the selected filter.")
        
        with tab8:
            st.subheader("🧤 Goalkeeper Specialized Clustering")
            st.write("Sub-clustering goalkeepers based on Save and Distribution stats")

            def cluster_goalkeepers_only(df):
                
                if 'Att (GK)' not in df.columns:
                
                    gk_df = df[df['Pos'].str.contains('GK', na=False, case=False)].copy()
                else:
                    gk_df = df[df['Att (GK)'] > 0].copy()
                
                if gk_df.empty:
                    return gk_df, [], None

                
                gk_features = [
                    'Att (GK)',     # GK involvement
                    'Thr',          # throws
                    'Launch%',      # long distribution tendency
                    'AvgLen',       # distribution length
                    'Stp',          # crosses stopped
                    'Stp%',         # efficiency
                    '#OPA',         # sweeping actions
                    '#OPA/90',
                    'AvgDist',      # defensive coverage
                    'Min'           # match involvement
                ]
                
                available_gk_features = [f for f in gk_features if f in gk_df.columns]
                
                if len(available_gk_features) < 2:
                    return pd.DataFrame(), [], None

                X_gk = gk_df[available_gk_features].fillna(0)
                scaler = StandardScaler()
                X_gk_scaled = scaler.fit_transform(X_gk)
                
                # K-Means clustering with 3 clusters as specified
                kmeans_gk = KMeans(n_clusters=3, random_state=42, n_init=10)
                gk_df['gk_cluster'] = kmeans_gk.fit_predict(X_gk_scaled)
                
                # Label mapping as specified
                gk_label_map = {
                    0: 'Elite Goalkeepers',
                    1: 'Regular First-Team Goalkeepers',
                    2: 'Backup / Reserve Goalkeepers'
                }
                gk_df['GK_Performance_Level'] = gk_df['gk_cluster'].map(gk_label_map)
                
                # PCA as specified
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_gk_scaled)
                gk_df['PCA1'] = X_pca[:, 0]
                gk_df['PCA2'] = X_pca[:, 1]
                
                # Profiling logic
                gk_profiles = (
                    gk_df
                    .groupby('GK_Performance_Level')[available_gk_features]
                    .mean()
                    .round(2)
                )
                
                return gk_df, available_gk_features, gk_profiles
    
                
            gk_only_df, gk_feats, gk_profiles = cluster_goalkeepers_only(df_filtered)
                
            if not gk_only_df.empty:
                col_gk_1, col_gk_2 = st.columns([2, 1])
                
                with col_gk_1:
                    # Interactive PCA for GK Performance Level
                    fig_gk_perf = px.scatter(
                        gk_only_df, x='PCA1', y='PCA2',
                        color='GK_Performance_Level',
                        hover_data=['Player', 'Squad'] + gk_feats,
                        title="Goalkeeper Performance Distribution (PCA)",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        height=500
                    )
                    st.plotly_chart(fig_gk_perf, use_container_width=True)
                
                with col_gk_2:
                    st.write("**Average Characteristics by Cluster**")
                    st.dataframe(gk_profiles, use_container_width=True)
                
                # Profile Bar Chart
                st.write("**Performance Characteristics Comparison**")
                fig_gk_bar = px.bar(
                    gk_profiles.reset_index().melt(id_vars='GK_Performance_Level'),
                    x='GK_Performance_Level', y='value', color='variable',
                    barmode='group', height=400,
                    title="Mean Performance Stats by Level"
                )
                st.plotly_chart(fig_gk_bar, use_container_width=True)
                
                st.write("**Goalkeeper Classification List**")
                st.dataframe(gk_only_df[['Player', 'Squad', 'GK_Performance_Level'] + gk_feats], use_container_width=True)
            else:
                st.warning("Att (GK)' stats or goalkeeper data are incomplete in the dataset.")
        
else:
    st.markdown("""
        <div class="welcome-box">
            <h2>👋 Football Analytics Dashboard: Insights at a Glance!</h2>
            <p>Please upload your <b>CSV Dataset</b> using the sidebar to get started.</p>
        </div>
    """, unsafe_allow_html=True)
