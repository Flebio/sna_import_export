import json
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import csv
from collections import defaultdict
from collections import Counter
from plotly.subplots import make_subplots


# Function to format GDP values
def format_gdp(gdp_value):
    try:
        if 'trillion' in gdp_value:
            return float(gdp_value.replace('$', '').replace(' trillion', '').replace(',', ''))
        elif 'billion' in gdp_value:
            return float(gdp_value.replace('$', '').replace(' billion', '').replace(',', '')) / 1000
        elif 'million' in gdp_value:
            return float(gdp_value.replace('$', '').replace(' million', '').replace(',', '')) / 1_000_000
        else:
            return None
    except ValueError:
        return None
        
def convert_to_decimal(coord):
    def parse_dms(dms):
        parts = dms.split()
        degrees = float(parts[0])
        minutes = float(parts[1])
        direction = parts[2]
        decimal = degrees + minutes / 60
        if direction in ['S', 'W']:
            decimal = -decimal
        return decimal

    lat, lon = coord.split(', ')
    lat_decimal = parse_dms(lat)
    lon_decimal = parse_dms(lon)
    return {"lat": lat_decimal, "lon": lon_decimal}

def create_df(data, country_mapping):
    partner_data = []
    node_attributes = {}
    
    for country, country_data in data.items():
        partners_export = country_data.get('Export Partners', {})
        partners_import = country_data.get('Import Partners', {})
        commodities_export = country_data.get('Export Commodities', '')
        commodities_import = country_data.get('Import Commodities', '')
        
        # Apply country mapping to country name
        country_mapped = country_mapping.get(country, country)
        
        # Extract GDP values based on the specified attributes
        gdp = format_gdp(country_data.get('GDP', ''))
        exports_money = format_gdp(country_data.get('Exports $', ''))
        imports_money = format_gdp(country_data.get('Imports $', ''))
        exports_gdp = country_data.get('GDP% Exports', '')
        imports_gdp = country_data.get('GDP% Imports', '')
        coords = country_data.get('Coordinates', {})
        
        # Populate node attributes based on the specified graph type
        node_attributes[country_mapped] = {
            'GDP': gdp,
            'Exports $': exports_money,
            'Exports GDP%': exports_gdp,
            'Imports $': imports_money,
            'Imports GDP%': imports_gdp,
            'Export Commodities': [commodity.strip() for commodity in commodities_export.split(',')],
            'Import Commodities': [commodity.strip() for commodity in commodities_import.split(',')],
            'x': coords.get('lon'),  # Longitude
            'y': coords.get('lat')   # Latitude
        }

        # Add export edges
        for partner, weight in partners_export.items():
            partner_mapped = country_mapping.get(partner, partner)
            partner_data.append({
                'source': country_mapped,
                'target': partner_mapped,
                'weight': float(weight),
                'type': 'Export'
            })
        
        # Add import edges (reversed direction)
        for partner, weight in partners_import.items():
            partner_mapped = country_mapping.get(partner, partner)
            partner_data.append({
                'source': partner_mapped,
                'target': country_mapped,
                'weight': float(weight),
                'type': 'Import'
            })
    
    df = pd.DataFrame(partner_data)
    return df, node_attributes

def create_graph(df, node_attributes):
    G = nx.DiGraph()
    
    # Add edges with type attribute
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'], type=row['type'])
    
    # Add node attributes
    for node, attributes in node_attributes.items():
        G.nodes[node].update(attributes)
    
    return G

def plot_network_on_world_map(G, centrality, cliques=None, title='Network Visualization', color_export='skyblue', color_import='orange'):
    edges_to_highlight = set()
    nodes_to_highlight = set()

    if cliques:
        for node in cliques:
            nodes_to_highlight.add(node)
        for edge in G.edges():
            if edge[0] in nodes_to_highlight and edge[1] in nodes_to_highlight:
                edges_to_highlight.add(edge)

    edge_trace_export = go.Scattergeo(
        locationmode='ISO-3',
        lon=[],
        lat=[],
        mode='lines',
        line=dict(width=1, color=color_export),
        hoverinfo='none'
    )

    edge_trace_import = go.Scattergeo(
        locationmode='ISO-3',
        lon=[],
        lat=[],
        mode='lines',
        line=dict(width=1, color=color_import),
        hoverinfo='none'
    )

    highlighted_edge_trace = go.Scattergeo(
        locationmode='ISO-3',
        lon=[],
        lat=[],
        mode='lines',
        line=dict(width=2, color='red'),
        hoverinfo='none'
    )

    for edge in G.edges(data=True):
        if 'x' in G.nodes[edge[0]] and 'y' in G.nodes[edge[0]] and 'x' in G.nodes[edge[1]] and 'y' in G.nodes[edge[1]]:
            x0, y0 = G.nodes[edge[0]]['x'], G.nodes[edge[0]]['y']
            x1, y1 = G.nodes[edge[1]]['x'], G.nodes[edge[1]]['y']
            if edge[:2] in edges_to_highlight:
                highlighted_edge_trace['lon'] += (x0, x1, None)
                highlighted_edge_trace['lat'] += (y0, y1, None)
            else:
                if edge[2]['type'] == 'Export':
                    edge_trace_export['lon'] += (x0, x1, None)
                    edge_trace_export['lat'] += (y0, y1, None)
                elif edge[2]['type'] == 'Import':
                    edge_trace_import['lon'] += (x0, x1, None)
                    edge_trace_import['lat'] += (y0, y1, None)

    max_centrality = max(centrality.values())
    min_centrality = min(centrality.values())

    node_trace = go.Scattergeo(
        locationmode='ISO-3',
        lon=[G.nodes[node]['x'] for node in G.nodes() if 'x' in G.nodes[node] and 'y' in G.nodes[node]],
        lat=[G.nodes[node]['y'] for node in G.nodes() if 'x' in G.nodes[node] and 'y' in G.nodes[node]],
        text=[node for node in G.nodes() if 'x' in G.nodes[node] and 'y' in G.nodes[node]],
        mode='markers+text',
        marker=dict(
            size=[10 + 20 * (centrality[node] - min_centrality) / (max_centrality - min_centrality) for node in G.nodes() if 'x' in G.nodes[node] and 'y' in G.nodes[node]],
            color=['red' if node in nodes_to_highlight else 'blue' for node in G.nodes() if 'x' in G.nodes[node] and 'y' in G.nodes[node]],
            line=dict(width=2, color='black')
        ),
        textfont=dict(
            size=7  # Set the font size of the labels here
        )
    )

    fig = go.Figure(data=[edge_trace_export, edge_trace_import, highlighted_edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        showlegend=False,
                        geo=dict(
                            scope='world',
                            projection_type='equirectangular',
                            showland=True,
                            subunitwidth=1,
                            countrywidth=1,
                            landcolor='rgb(217, 217, 217)',
                            coastlinecolor='rgb(255, 255, 255)',
                            subunitcolor='rgb(255, 255, 255)',
                            countrycolor='rgb(255, 255, 255)',
                            showcountries=True,
                            showocean=True,
                            showlakes=True,
                            lonaxis=dict(
                                range=[-180, 180]
                            ),
                            lataxis=dict(
                                range=[-90, 90]
                            ),
                            resolution=50  # Make map more detailed
                        ),
                        width=1000,  # Adjust the width of the plot
                        height=600  # Adjust the height of the plot
                    )
                )
    fig.show()

# Plot Centrality Histogram
def centrality(cent, type, threshold, color='skyblue'):
    # Sort countries and their centrality values alphabetically by country names
    sorted_items = sorted(cent.items())
    sorted_countries = [item[0] for item in sorted_items]
    sorted_centrality_values = [item[1] for item in sorted_items]

    fig = go.Figure(data=[go.Bar(
        x=sorted_countries,
        y=sorted_centrality_values,
        marker_color=color
    )])

    fig.update_layout(
        title=type + ' of Nodes',
        xaxis_title='Country',
        yaxis_title=type,
        xaxis=dict(tickangle=270, tickfont=dict(size=3)),
        height=500
    )

    fig.show()

    # Find nodes with high centrality
    high_centrality_nodes = {node: centrality for node, centrality in cent.items() if centrality > threshold}

    # Sort high centrality nodes by their centrality values in descending order
    sorted_high_centrality_nodes = sorted(high_centrality_nodes.items(), key=lambda item: item[1], reverse=True)

    print('Nodes (Countries) with High ' + type + ' (> ' + str(threshold) + '):')
    for node, centrality in sorted_high_centrality_nodes:
        print(f"{node}: {centrality:.4f}")

    return dict(sorted_high_centrality_nodes)
   
def centrality_heatmap(cent, type, color='Viridis'):
    countries = list(cent.keys())
    centrality_values = list(cent.values())

    fig = go.Figure(data=go.Choropleth(
        locations=countries,
        locationmode='country names',
        z=centrality_values,
        colorscale=color
    ))

    fig.update_layout(
        title=f'{type} Heatmap',
        geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
        height=500,
        width=700
    )

    fig.show()


def top_centrality(cent, type, top_n=15, color='skyblue'):
    # Sort countries and their centrality values by centrality values in descending order
    sorted_items = sorted(cent.items(), key=lambda item: item[1], reverse=True)
    
    # Take only the top_n items
    top_items = sorted_items[:top_n]
    sorted_countries = [item[0] for item in top_items][::-1]  # Reverse the order
    sorted_centrality_values = [round(item[1], 3) for item in top_items][::-1]  # Reverse the order and round to 3 decimals

    fig = go.Figure(data=[go.Bar(
        y=sorted_countries,  # Use y instead of x for horizontal bars
        x=sorted_centrality_values,  # Use x instead of y for horizontal bars
        marker_color=color,
        orientation='h',  # Set the orientation to horizontal
        text=sorted_centrality_values,  # Add text to bars
        textposition='auto'  # Position text automatically for better visibility
    )])

    fig.update_layout(
        title=f'Top {top_n} {type} of Nodes',
        xaxis_title=type,
        yaxis_title='Country',
        xaxis=dict(tickfont=dict(size=10)),  # Adjusted font size for better visibility
        width=500,  # Set width to 500
        height=500  # Set height to 500
    )

    fig.show()

# Plot top n countries with the biggest differential between two centrality values
def top_centrality_diff(cent1, cent2, type1, type2, typello, top_n=15, color1='skyblue', color2='orange'):
    
    diff = {country: cent1.get(country, 0) - cent2.get(country, 0) for country in set(cent1) | set(cent2)}

    sorted_diff_items = sorted(diff.items(), key=lambda item: abs(item[1]), reverse=True)
    
    top_diff_items = sorted_diff_items[:top_n]
    sorted_countries = [item[0] for item in top_diff_items][::-1] 
    #sorted_differential_values = [round(item[1], 3) for item in top_diff_items][::-1]  

    cent1_values = [round(cent1.get(country, 0), 3) for country in sorted_countries]
    cent2_values = [round(cent2.get(country, 0), 3) for country in sorted_countries]

    
    trace1 = go.Bar(
        y=sorted_countries, 
        x=cent1_values, 
        marker_color=color1,
        orientation='h', 
        name=type1,
        text=cent1_values,
        textposition='auto'
    )

    trace2 = go.Bar(
        y=sorted_countries,  
        x=cent2_values, 
        marker_color=color2,
        orientation='h', 
        name=type2,
        text=cent2_values,  
        textposition='auto'  
    )

    fig = go.Figure(data=[trace1, trace2])

    fig.update_layout(
        title=f'Top {top_n} Countries with Biggest Differential in {typello}',
        xaxis_title='Centrality Value',
        yaxis_title='Country',
        xaxis=dict(tickfont=dict(size=10)),  
        width=700, 
        height=600,  
        barmode='group'  
    )

    fig.show()

# Plot Centrality Power Law
def plot_centrality_power_law(centrality, bins, type, color):
    centrality_values = list(centrality.values())
    # Calculate histogram
    freq, bins = np.histogram(centrality_values, bins=bins)

    # Calculate bin centers
    x = (bins[:-1] + bins[1]) / 2  # x = center value of each bin
    y = freq  # y = occurrence

    # Filter out zero values
    non_zero_indices = np.where(y > 0)
    x_display = x[non_zero_indices]
    y_display = y[non_zero_indices]

    # Exclude the last value for the fit (if it's an outlier)
    fit_points = np.where(x_display < np.max(x_display))
    x_fit = x_display[fit_points]
    y_fit = y_display[fit_points]

    # Take the logarithm of x and y
    log_x_display = np.log10(x_display)
    log_y_display = np.log10(y_display)
    log_x_fit = np.log10(x_fit)
    log_y_fit = np.log10(y_fit)

    # Fit a straight line to the data
    coeffs = np.polyfit(log_x_fit, log_y_fit, 1)

    # Generate y-values for the fitted line
    fitted_y = coeffs[0] * log_x_fit + coeffs[1]

    # Perform linear regression and get p-value
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x_fit, log_y_fit)

    # Plot the data and the fit
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=log_x_display,
        y=log_y_display,
        mode='markers',
        name='Data',
        marker=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=log_x_fit,
        y=fitted_y,
        mode='lines',
        name='Best fit line',
        line=dict(color=color)
    ))

    fig.update_layout(
        title='Log-Log Plot of ' + type + ' Centrality',
        xaxis_title='Log ' + type + ' Centrality',
        yaxis_title='Log Frequency',
        height=500,
        annotations=[dict(
            x=min(log_x_display),
            y=max(log_y_display),
            text=f'p-value: {p_value:.1e} \nR: {r_value:.2f}',
            showarrow=False,
            bgcolor='white'
        )]
    )

    fig.show()

    print(f"The slope of the line is: {slope}")
    return slope

# Plot Cumulative Distribution Graph
def cumulative_distribution(centrality, type, color):
    # Sort the degree centrality values
    centrality_values = list(centrality.values())
    sorted_centrality_values = np.sort(centrality_values)

    # Cumulative distribution
    cdf = np.arange(1, len(sorted_centrality_values) + 1) / len(sorted_centrality_values)

    # CDF Plot
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Linear Scale', 'Log-Log Scale'))

    # Linear scale plot
    fig.add_trace(go.Scatter(
        x=sorted_centrality_values,
        y=cdf,
        mode='lines',
        name='Linear',
        line=dict(color=color)
    ), row=1, col=1)

    # Log-log scale plot
    fig.add_trace(go.Scatter(
        x=sorted_centrality_values,
        y=cdf,
        mode='lines',
        name='Log-Log',
        line=dict(color=color)
    ), row=1, col=2)

    fig.update_xaxes(title_text=type + ' Centrality', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Frequency', row=1, col=1)
    fig.update_xaxes(type="log", title_text=type + ' Centrality', row=1, col=2)
    fig.update_yaxes(type="log", title_text='Cumulative Frequency', row=1, col=2)

    fig.update_layout(
        title=type + ' Centrality Cumulative Distribution',
        height=500
    )

    fig.show()


    # Pearson correlation coefficient
def correlation_cd(clustering_coeffs, degree_centrality, color='skyblue'):
    clustering_coeffs_values = list(clustering_coeffs.values())
    degree_centrality_values = list(degree_centrality.values())
    correlation, p_value = stats.pearsonr(clustering_coeffs_values, degree_centrality_values)

    print(f"Pearson correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4e}")

    # Prepare data for Plotly
    data = {
        'Degree Centrality': degree_centrality_values,
        'Local Clustering Coefficient': clustering_coeffs_values
    }

    # Plotting the correlation
    fig = px.scatter(
        data,
        x='Degree Centrality',
        y='Local Clustering Coefficient',
        title=f'Correlation between Degree Centrality and Local Clustering Coefficient\nPearson r = {correlation:.4f}',
        labels={'x': 'Degree Centrality', 'y': 'Local Clustering Coefficient'},
        color_discrete_sequence=[color]
    )

    fig.update_layout(height=500)

    fig.show()
    return correlation, p_value

def kcores(G, a, b):
    k_values = range(a, b)  # Define the k values
    k_core_nodes_dict = {}  # Dictionary to store k-core nodes

    for k in k_values:
        k_core_subgraph = nx.k_core(G, k)
        k_core_nodes = set(k_core_subgraph.nodes())

        if len(k_core_nodes) > 0:  # Only store non-empty k-cores
            k_core_nodes_dict[k] = k_core_nodes
            print(f"\nNodes in the {k}-core ({len(k_core_nodes)} elements):", sorted(k_core_nodes))

    # Find the highest k-core with nodes
    highest_k = max(k_core_nodes_dict.keys(), default=None)
    if highest_k is not None:
        most_high_kcore = k_core_nodes_dict[highest_k]
    else:
        most_high_kcore = set()

    return k_core_nodes_dict, most_high_kcore

# Plot the Degree Correlation
def deg_cor(G, color='skyblue'):
    # Compute the degree of each node
    degree = dict(G.degree())

    # Compute the average neighbor degree for each node
    avg_neighbor_degree = nx.average_neighbor_degree(G)

    # Prepare data for plotting
    degrees = list(degree.values())
    avg_neighbor_degrees = [avg_neighbor_degree[node] for node in G.nodes()]

    # Scatter plot
    fig = px.scatter(
        x=degrees,
        y=avg_neighbor_degrees,
        labels={'x': 'Node Degree', 'y': 'Average Neighbor Degree'},
        title='Degree Correlation',
        color_discrete_sequence=[color]
    )

    fig.update_layout(height=500)

    fig.show()

# Extract data for histograms
def extract_node_attributes(graph, attribute):
    return {node: data.get(attribute, 0) for node, data in graph.nodes(data=True)}

def find_countries_with_common_commodities(G):
    commodity_to_countries = defaultdict(list)
    commodity_to_countries_export = defaultdict(list)
    commodity_to_countries_import = defaultdict(list)
    
    for country, data in G.nodes(data=True):
        export_commodities = data.get('Export Commodities', [])
        import_commodities = data.get('Import Commodities', [])
        # Use a set to avoid duplicates
        commodities = set(export_commodities) | set(import_commodities)
        
        for commodity in commodities:
            commodity_to_countries[commodity].append(country)
        for commodity in export_commodities:
            commodity_to_countries_export[commodity].append(country)
        for commodity in import_commodities:
            commodity_to_countries_import[commodity].append(country)
    
    return commodity_to_countries, commodity_to_countries_export, commodity_to_countries_import


def analyze_commodities(node_attributes):
    commodity_counter_export = Counter()
    commodity_counter_import = Counter()
    for country, attributes in node_attributes.items():
        commodities_export = attributes.get('Export Commodities', [])
        commodity_counter_export.update(commodities_export)
        commodities_import = attributes.get('Import Commodities', [])
        commodity_counter_import.update(commodities_import)
    
    return commodity_counter_export, commodity_counter_import

def plot_attributes(data_dict, title, xlabel, ylabel, color='skyblue', colorscale='Viridis'):
    # Sort the data by country names (keys)
    sorted_items = sorted(data_dict.items())
    sorted_countries = [item[0] for item in sorted_items]
    sorted_values = [item[1] for item in sorted_items]

    # Create and display the bar chart
    bar_fig = go.Figure(data=[go.Bar(
        x=sorted_countries,
        y=sorted_values,
        marker_color=color
    )])

    bar_fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis=dict(tickangle=270, tickfont=dict(size=3)),
        height=500
    )

    bar_fig.show()

    # Create and display the world heatmap
    heatmap_fig = go.Figure(data=go.Choropleth(
        locations=sorted_countries,
        locationmode='country names',
        z=sorted_values,
        colorscale=colorscale,
        colorbar_title=ylabel
    ))

    heatmap_fig.update_layout(
        title=title + ' World Heatmap',
        height=500
    )

    heatmap_fig.show()

def plot_common_commodities_histogram(commodity_to_countries, name='Network', color='skyblue'):
    commodity_counts = {commodity: len(countries) for commodity, countries in commodity_to_countries.items()}
    sorted_items = sorted(commodity_counts.items(), key=lambda item: item[1], reverse=True)
    commodities = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    fig = go.Figure(data=[go.Bar(
        x=commodities,
        y=counts,
        marker_color=color
    )])

    fig.update_layout(
        title=f'Number of Countries Sharing Each Commodity in the {name}',
        xaxis_title='Commodity',
        yaxis_title='Number of Countries',
        xaxis=dict(tickangle=270, tickfont=dict(size=2)),
        height=500
    )

    fig.show()

def print_common_commodities(commodity_to_countries, type='Export'):
    for commodity, countries in commodity_to_countries.items():
        if len(countries) > 1:
            print(f"Commodity '{commodity}' is shared by countries: {', '.join(countries)}")

def plot_commodity_on_world_map(G, centrality, commodity, title='Commodity Sharing Network Visualization', colorline='skyblue'):
    nodes_to_highlight = set()
    edges_to_highlight = set()

    for country, data in G.nodes(data=True):
        export_commodities = data.get('Export Commodities', [])
        import_commodities = data.get('Import Commodities', [])
        commodities = set(export_commodities) | set(import_commodities)
        if commodity in commodities:
            nodes_to_highlight.add(country)

    for edge in G.edges():
        if edge[0] in nodes_to_highlight and edge[1] in nodes_to_highlight:
            edges_to_highlight.add(edge)

    edge_trace = go.Scattergeo(
        locationmode='ISO-3',
        lon=[],
        lat=[],
        mode='lines',
        line=dict(width=1, color=colorline),
        hoverinfo='none'
    )

    highlighted_edge_trace = go.Scattergeo(
        locationmode='ISO-3',
        lon=[],
        lat=[],
        mode='lines',
        line=dict(width=2, color='red'),
        hoverinfo='none'
    )

    for edge in G.edges():
        if 'x' in G.nodes[edge[0]] and 'y' in G.nodes[edge[0]] and 'x' in G.nodes[edge[1]] and 'y' in G.nodes[edge[1]]:
            x0, y0 = G.nodes[edge[0]]['x'], G.nodes[edge[0]]['y']
            x1, y1 = G.nodes[edge[1]]['x'], G.nodes[edge[1]]['y']
            if edge in edges_to_highlight:
                highlighted_edge_trace['lon'] += (x0, x1, None)
                highlighted_edge_trace['lat'] += (y0, y1, None)
            else:
                edge_trace['lon'] += (x0, x1, None)
                edge_trace['lat'] += (y0, y1, None)

    max_centrality = max(centrality.values())
    min_centrality = min(centrality.values())

    node_trace = go.Scattergeo(
        locationmode='ISO-3',
        lon=[G.nodes[node]['x'] for node in G.nodes() if 'x' in G.nodes[node] and 'y' in G.nodes[node]],
        lat=[G.nodes[node]['y'] for node in G.nodes() if 'x' in G.nodes[node] and 'y' in G.nodes[node]],
        text=[node for node in G.nodes() if 'x' in G.nodes[node] and 'y' in G.nodes[node]],
        mode='markers+text',
        marker=dict(
            size=[10 + 20 * (centrality[node] - min_centrality) / (max_centrality - min_centrality) for node in G.nodes() if 'x' in G.nodes[node] and 'y' in G.nodes[node]],
            color=['red' if node in nodes_to_highlight else 'blue' for node in G.nodes() if 'x' in G.nodes[node] and 'y' in G.nodes[node]],
            line=dict(width=2, color='black')
        ),
        textfont=dict(
            size=7  # Set the font size of the labels here
        )
    )

    fig = go.Figure(data=[edge_trace, highlighted_edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        showlegend=False,
                        geo=dict(
                            scope='world',
                            projection_type='equirectangular',
                            showland=True,
                            subunitwidth=1,
                            countrywidth=1,
                            landcolor='rgb(217, 217, 217)',
                            coastlinecolor='rgb(255, 255, 255)',
                            subunitcolor='rgb(255, 255, 255)',
                            countrycolor='rgb(255, 255, 255)',
                            showcountries=True,
                            showocean=True,
                            showlakes=True,
                            lonaxis=dict(
                                range=[-180, 180]
                            ),
                            lataxis=dict(
                                range=[-90, 90]
                            ),
                            resolution=50  # Make map more detailed
                        ),
                        width=1000,  # Adjust the width of the plot
                        height=600  # Adjust the height of the plot
                    )
                )
    fig.show()


