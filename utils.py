import json, csv
import networkx as nx
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
from collections import Counter

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# We retrieve only the countries that have all the followings informations
key_mapping = {
        "Economy: Real GDP (purchasing power parity)": "GDP",
        "Economy: Exports - partners": "Export Partners",
        "Economy: Exports - commodities": "Export Commodities",
        "Economy: Imports - partners": "Import Partners",
        "Economy: Imports - commodities": "Import Commodities",
        "Geography: Geographic coordinates": "Coordinates",
        "Government: Government type": 'Government type'
    }

# And map some country names for convenience
country_mapping = {
    'UK': 'United Kingdom',
    'US': 'United States',
    'UAE': 'United Arab Emirates',
    'Gambia, The': 'Gambia',
    'Turkey (Turkiye)': 'Turkey',
    'Congo, Democratic Republic of the': 'Congo',
    'Democratic Republic of the Congo': 'Congo',
    'Congo, Republic of the': 'Congo',
    'Republic of the Congo': 'Congo',
    'Korea, South': 'South Korea',
    'Korea, North': 'North Korea',
    'Cost Rica': 'Costa Rica',
    'NZ': 'New Zealand',
    'Micronesia, Federated States of': 'Micronesia',
    'Bahamas, The': 'Bahamas'
}

# We use another taxonomy for the government types, joining the values in standardized macrocategories
government_type_mapping = {
    'presidential republic': 'Presidential Republic',
    'Republic of Cyprus - presidential republic; self-declared "Turkish Republic of Northern Cyprus"': 'Presidential Republic',
    'presidential republic in free association with the US': 'Presidential Republic',
    
    'constitutional federal republic': 'Federal Republic',
    'federal republic in free association with the US': 'Federal Republic',
    'federal republic': 'Federal Republic',
    
    'federal presidential republic': 'Federal Presidential Republic',
    
    'federal parliamentary democracy': 'Federal Parliamentary Republic',
    'federal parliamentary republic': 'Federal Parliamentary Republic',

    'parliamentary democracy under a constitutional monarchy; a Commonwealth realm': 'Constitutional Monarchy',
    'federal parliamentary democracy under a constitutional monarchy; a Commonwealth realm': 'Constitutional Monarchy',
    'federal parliamentary democracy under a constitutional monarchy': 'Constitutional Monarchy',
    'parliamentary constitutional monarchy': 'Constitutional Monarchy',
    'parliamentary republic; a Commonwealth realm': 'Constitutional Monarchy',
    'parliamentary democracy; part of the Kingdom of the Netherlands': 'Constitutional Monarchy',
    'Overseas Territory of the UK with limited self-government; parliamentary democracy': 'Constitutional Monarchy',
    'parliamentary constitutional monarchy; a Commonwealth realm': 'Constitutional Monarchy',
    'parliamentary democracy; self-governing overseas territory of the UK': 'Constitutional Monarchy',
    'constitutional monarchy': 'Constitutional Monarchy',
    'federal parliamentary constitutional monarchy': 'Constitutional Monarchy',
    'parliamentary constitutional monarchy; part of the Kingdom of the Netherlands': 'Constitutional Monarchy',
    'parliamentary democracy under a constitutional monarchy': 'Constitutional Monarchy',

    'unincorporated organized territory of the US with local self-government; republican form of territorial government with separate executive, legislative, and judicial branches': 'Territorial Government',
    'unincorporated, unorganized Territory of the US with local self-government; republican form of territorial government with separate executive, legislative, and judicial branches': 'Territorial Government',
    'a commonwealth in political union with and under the sovereignty of the US; republican form of government with separate executive, legislative, and judicial branches': 'Territorial Government',
    'unincorporated organized territory of the US with local self-government; republican form of territorial government with separate executive, legislative, and judicial branches; note - reference Puerto Rican Federal Relations Act, 2 March 1917, as amended by Public Law 600, 3 July 1950': 'Territorial Government',

    'presidential republic; authoritarian': 'Authoritarian Republic',
    'presidential republic; highly authoritarian': 'Authoritarian Republic',
    'presidential republic; highly authoritarian regime': 'Authoritarian Republic',

    'executive-led limited democracy; a special administrative region of the People\'s Republic of China': 'Limited Democracy',
    'presidential limited democracy; a special administrative region of the People\'s Republic of China': 'Limited Democracy',

    'presidential republic in name, although in fact a dictatorship': 'Dictatorship',
    'dictatorship, single-party state; official state ideology of "Juche" or "national self-reliance"': 'Dictatorship',

    'parliamentary republic': 'Parliamentary Republic',
    'unitary parliamentary republic': 'Parliamentary Republic',
    'parliamentary democracy; note - constitutional changes adopted in December 2015 transformed the government to a parliamentary system': 'Parliamentary Republic',
    'parliamentary democracy': 'Parliamentary Republic',

    'theocratic; the United States does not recognize the Taliban Government': 'Theocratic',
    'theocratic republic': 'Theocratic',

    'communist state': 'Communist State',
    'communist party-led state': 'Communist State',

    'mixed presidential-parliamentary system in free association with the US': 'Mixed System',

    'absolute monarchy or sultanate': 'Absolute Monarchy',
    'absolute monarchy': 'Absolute Monarchy',

    'federation of monarchies': 'Federation of Monarchies',
    
    'semi-presidential republic': 'Semi-Presidential Republic',
    'semi-presidential federation': 'Semi-Presidential Federation',
    
    'military regime': 'Military Regime',

    'in transition': 'Transitional Government'
}

# We use another taxonomy for the commodities as well, joining the values in standardized macrocategories
sitc_hierarchy = {
    "0 - Food and live animals": {
        "00 - Live animals": ["sheep", "goats", "sheep and goats", "cattle"],
        "01 - Meat and meat preparations": ["beef", "pork", "poultry meats", "mutton"],
        "02 - Dairy products and birds' eggs": ["milk", "milk cream powders", "butter", "cheese", "concentrated milk"],
        "03 - Fish, crustaceans, mollusks and aquatic invertebrates": [
            "fish", "fish fillets", "frozen fish", "mackerel", "shrimp", "tuna", "salmon", 
            "skipjack", "sardines", "crabs", "lobsters and crawfish", "mollusks", "shellfish", 
            "mollusks and shellfish", "cod", "halibut", "eels", "other processed and frozen fish", 
            "fish and fish products", "crustaceans", "lobster", "skipjack and tuna", "skipjack and stripe-bellied bonito"
        ],
        "04 - Cereals and cereal preparations": ["wheat", "rice", "corn", "wheat flours", "durum wheat"],
        "05 - Vegetables and fruit": [
            "bananas", "grapes", "pineapples", "mangoes", "apples", "pears", "onions", "fruits", 
            "peas", "tomatoes", "lemons", "pitted fruits", "dried fruits", "tropical fruits", "ground nuts", 
            "cashews", "figs", "nuts", "fruits and nuts", "various vegetables", "various fruits and nuts", "vegetables", "kidney beans", 
            "dried legumes", "legumes", "cassava", "cumin seeds", "pistachios", "dried legumes and nuts"
        ],
        "06 - Sugars, sugar preparations and honey": ["raw sugar", "sugar"],
        "07 - Coffee, tea, cocoa, spices, and manufactures thereof": [
            "coffee", "tea", "cocoa beans", "cocoa products", "cocoa paste", 
            "vanilla", "cloves", "nutmeg", "spices", "sauces and seasonings", "chocolate"
        ],
        "08 - Feeding stuff for animals": ["animal meal", "animal food", "fishmeal", "forage crops"],
        "09 - Miscellaneous edible products and preparations": ["food preparations", "foodstuffs"]
    },
    "1 - Beverages and tobacco": {
        "11 - Beverages": ["beer", "wine", "liquors", "whiskies", "rum", "rums and liquors", "rums", "hard liquors", 
                           "carbonated drink mixtures", "flavored water", "water", "mineral water"],
        "12 - Tobacco and tobacco manufactures": ["tobacco", "cigarettes", "cigars"]
    },
    "2 - Crude materials, inedible, except fuels": {
        "22 - Oil seeds and oleaginous fruits": ["soybeans", "soybean products", "soy products", "soybeans and soybean products", 
                                                 "rapeseed", "sesame seeds", "sunflower seeds"
        ],
        "23 - Crude rubber": ["rubber", "vulcanized rubber"],
        "24 - Cork and wood": ["lumber", "fuel wood", "fuel woods"],
        "25 - Pulp and waste paper": ["sulfate wood pulp", "wood pulp"],
        "26 - Textile fibers and their wastes": [
            "cotton", "wool", "cotton fibers", "animal hair"
        ],
        "27 - Crude fertilizers and crude minerals": [
             "gravel", "sand", "dolomite", "gypsum", "manganese"
        ],
        "28 - Metalliferous ores and metal scrap": [
            "copper ore", "scrap aluminum", "scrap copper", "iron ore and reductions", 
            "aluminum ores and oxide", "iron ores", "iron ore", "raw earth metal ores", "scrap iron"
        ],
        "29 - Crude animal and vegetable materials": [
             "copra", "insect resins", "gum arabic", "coral and shells", "cut flowers"
        ]
    },
    "3 - Mineral fuels, lubricants and related materials": {
        "32 - Coal, coke and briquettes": ["coal", "coke", "coal tar oil"],
        "33 - Petroleum, petroleum products and related materials": [
            "refined petroleum", "crude petroleum", "petroleum coke", "crude and refined petroleum"
        ],
        "34 - Gas, natural and manufactured": ["natural gas", "propane", "butane"],
        "35 - Electric current": ["electricity"]
    },
    "4 - Animal and vegetable oils, fats and waxes": {
        "41 - Animal oils and fats": [],
        "42 - Fixed vegetable oils and fats": [
            "coconut oil", "palm oil", "olive oil", "soybean oil", "industrial fatty acids/oils", "sunflower seed oils"
        ],
        "43 - Animal or vegetable fats and oils, processed; waxes of animal or vegetable origin, inedible mixtures or preparations of animal or vegetable fats or oils, n.e.s.": []
    },
    "5 - Chemicals and related products, n.e.s.": {
        "51 - Organic chemicals": ["industrial hydrocarbons", "nitrile compounds", "nitrogen compounds", "industrial hydrocarbon products"],
        "52 - Inorganic chemicals": ["chlorides", "halogens", "hydrogen", "industrial chemical liquids", "laboratory reagents", 
                                     "sulfur compounds", "ammonia", "aluminum oxide"
        ],
        "53 - Dyeing, tanning and coloring materials": ["acrylic paints"],
        "54 - Medicinal and pharmaceutical products": [
            "packaged medicines", "vaccines and cultures", "medical cultures and vaccines", "medical cultures/vaccines",
            "dental filling pharmaceuticals", "hormones", "surgical instruments", "medical instruments"
        ],
        "55 - Essential oils and resinoids and perfume materials; toilet, polishing and cleansing preparations": [
            "scented mixtures", "beauty products", "soap", "bandages", "sutures", "essential oils", "perfume plants"
        ],
        "56 - Fertilizers": ["fertilizers", "nitrogen fertilizers"],
        "57 - Plastics in primary forms": ["polymers", "ethylene polymers", "styrene polymers", "building plastics"],
        "58 - Plastics in non-primary forms": ["plastic lids", "adhesive plastics", "plastic building materials"],
        "59 - Chemical materials and products": [
            "industrial additives", "phosphoric acid", "industrial alcohols", 
            "calcium phosphates", "support catalysts"
        ]
    },
    "6 - Manufactured goods classified chiefly by material": {
        "61 - Leather, leather manufactures and dressed furskins": [
            "leather footwear", "leather footwear and parts", "leather goods", "leather handbags"
        ],
        "62 - Rubber manufactures": ["rubber apparel", "tires"],
        "63 - Cork and wood manufactures": ["veneer sheeting"],
        "64 - Paper, paperboard and articles of paper pulp, of paper or of paperboard": [
            "kaolin-coated paper", "paper", "paper labels", "toilet paper"
        ],
        "65 - Textile yarn, fabrics, made-up articles, n.e.s., and related products": ["textiles"],
        "66 - Non-metallic mineral manufactures": ["cement", "asphalt mixtures", "float glass", "gemstones", 
                                                           "precious stones", "pearls", "diamonds", "semiconductors"
        ],
        "67 - Iron and steel": ["iron", "iron products", "iron alloys", "steel products", "iron/iron products", "stainless steel", 
                                "iron structures"
        ],
        "68 - Non-ferrous metals": ["aluminum", "aluminum ores", "copper", "nickel", "zinc", "tin", "titanium", "lead", "antimony", 
                                    "platinum", "molybdenum", "aluminum and aluminum products", "silver", "scandium", "yttrium", "uranium",
                                    "precious metals"
        ],
        "69 - Manufactures of metals": ["metal piping", "aluminum wiring", "aluminum plating", "valves", "copper piping"]
    },
    "7 - Machinery and transport equipment": {
        "71 - Power generating machinery and equipment": ["electric generators", "gas turbines", "hydraulic engines", "spark-ignition engines"],
        "72 - Machinery specialized for particular industries": [
            "industrial washing/bottling machinery", "woodworking machinery", "boring/sinking machinery", 
            "drilling platforms", "mining machinery", "excavation machinery", "stone processing machinery", "air pumps", "stone grinders"
        ],
        "73 - Metalworking machinery": [],
        "74 - General industrial machinery and equipment and machine parts": [
            "industrial machinery", "general machinery",  "centrifuges"
        ],
        "75 - Office machines and automatic data processing machines": [
            "computers", "office machinery", "office machinery and parts", "office machinery/parts"
        ],
        "76 - Telecommunications and sound recording and reproducing apparatus and equipment": [
            "telephones", "broadcasting equipment", "video displays", "liquid crystal displays"
        ],
        "77 - Electrical machinery, apparatus and appliances and electrical parts": [
            "integrated circuits", "insulated wiring", "electric batteries", "electrical plugs and sockets", 
            "electrical panels", "low-voltage protection equipment", "electrical transformers", 
            "electrical capacitors", "electrical control boards", "electrical power accessories", 
            "electrical lighting/signaling equipment", "appliances", "transformers"
        ],
        "78 - Road vehicles": [
            "cars", "cars and vehicle parts", "cars/vehicle parts", "vehicle parts", "delivery trucks", 
            "motor vehicle parts/accessories", "trailers", "construction vehicles", "large construction vehicles"
        ],
        "79 - Other transport equipment": [
            "aircraft", "aircraft parts", "ships", "tug boats", "tugboats", "cargo ships", "fishing ships", 
            "recreational boats", "floating platforms", "railway shipping containers", "scrap vessels"
        ]
    },
    "8 - Miscellaneous manufactured articles": {
        "81 - Prefabricated buildings; sanitary plumbing, heating and lighting fixtures and fittings, n.e.s.": [
            "prefabricated buildings", "light fixtures"
        ],
        "82 - Furniture, bedding, mattresses, mattress supports, cushions and similar stuffed furnishings": [
            "furniture", "wood furniture", "wooden furniture", "mattress materials", "seating", "carpets", "seats"
        ],
        "83 - Travel goods, handbags and similar containers": ["trunks and cases"],
        "84 - Articles of apparel and clothing accessories": ["clothing", "clothing and apparel", "clothing products", "clothing and footwear"],
        "85 - Footwear": ["footwear"],
        "87 - Professional, scientific and controlling instruments and apparatus": [
            "measuring instruments", "photo lab equipment", "needles and catheters", "orthopedic and medical appliances", "orthopedic appliances"

        ],
        "88 - Photographic apparatus, equipment and supplies and optical goods; watches and clocks": [
            "watches", "photography equipment"
        ],
        "89 - Miscellaneous manufactured articles": [
            "children's toys and stuffed animals", "postage stamps", "collector's items", 
            "postage stamps and title bond paper documents", "jewelry", "paintings"
        ]
    },
    "9 - Commodities and transactions not classified elsewhere in SITC": {
        "97 - Gold, non-monetary": ["gold"]
    }
}

def parse_value(value_str):
    return int(value_str.replace('$', '').replace(',', ''))

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

def get_sitc_level_2(commodity, sitc_hierarchy):
    for level_2_dict in sitc_hierarchy.values():
        for level_2, commodities in level_2_dict.items():
            if commodity in commodities:
                return level_2 #.split(' - ')[0]  # Return only the Level 2 SITC code
    return None  # Return None if no match is found

def create_df(data, country_mapping):
    partner_data = []
    node_attributes = {}
    
    existing_edges = set()  # To track existing edges and avoid duplicates
    
    for country, country_data in data.items():
        gdp = country_data.get('GDP', '') 
        partners_export = country_data.get('Export Partners', {})
        partners_import = country_data.get('Import Partners', {})
        commodities_export = country_data.get('Export Commodities', '')
        commodities_export_sitc = country_data.get('Export Commodities SITC', '')
        # commodities_import = country_data.get('Import Commodities', '')
        coords = country_data.get('Coordinates', {})
        gov_type = country_data.get('Government type', {})
        
        country_mapped = country_mapping.get(country, country)        
        node_attributes[country_mapped] = {
            'Government Type': gov_type,
            'GDP': gdp,
            'Export Commodities': [commodity.strip() for commodity in commodities_export.split(',')],
            'Export Commodities SITC': commodities_export_sitc,
            # 'Import Commodities': [commodity.strip() for commodity in commodities_import.split(',')],
            'x': coords.get('lon'),  # Longitude
            'y': coords.get('lat')   # Latitude
        }

        # Add export edges
        for partner, _ in partners_export.items():
            partner_mapped = country_mapping.get(partner, partner)
            if partner_mapped in data:  # Only add the edge if partner exists in the data
                edge = (country_mapped, partner_mapped)
                if edge not in existing_edges:  # Avoid duplicate edges
                    partner_data.append({
                        'source': country_mapped,
                        'target': partner_mapped
                    })
                    existing_edges.add(edge)
        
        # Add import edges (reversed direction)
        for partner, _ in partners_import.items():
            partner_mapped = country_mapping.get(partner, partner)
            if partner_mapped in data:  # Only add the edge if partner exists in the data
                edge = (partner_mapped, country_mapped)
                if edge not in existing_edges:  # Avoid duplicate edges
                    partner_data.append({
                        'source': partner_mapped,
                        'target': country_mapped
                    })
                    existing_edges.add(edge)

    df = pd.DataFrame(partner_data)
    
    return df, node_attributes

def create_graph(df, node_attributes):
    G = nx.DiGraph()
    
    # Add edges with type attribute
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'])
    
    # Add node attributes
    for node, attributes in node_attributes.items():
        G.nodes[node].update(attributes)

    return G

def plot_network_on_world_map(G, centrality, cliques=None, title='Network Visualization', edges_color='skyblue', highlight_edges_color='lightcoral', markers_color = '#3366cc', highlight_markers_color = '#dc3912'):
    edges_to_highlight = set()
    nodes_to_highlight = set()

    if cliques:
        for node in cliques:
            nodes_to_highlight.add(node)
        for edge in G.edges():
            if edge[0] in nodes_to_highlight and edge[1] in nodes_to_highlight:
                edges_to_highlight.add(edge)

    edge_trace = go.Scattergeo(
        locationmode='ISO-3',
        lon=[],
        lat=[],
        mode='lines',
        line=dict(width=1, color=edges_color),
        hoverinfo='none'
    )
    
    highlighted_edge_trace = go.Scattergeo(
        locationmode='ISO-3',
        lon=[],
        lat=[],
        mode='lines',
        line=dict(width=2, color=highlight_edges_color),
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
            color=[highlight_markers_color if node in nodes_to_highlight else markers_color for node in G.nodes() if 'x' in G.nodes[node] and 'y' in G.nodes[node]],
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

def histograms(measure_dict, type, n=10, color='skyblue', results='both'):
    # Sort countries and their values in descending order
    sorted_items = sorted(measure_dict.items(), key=lambda item: -item[1])
    
    # Extract the top n and worst n countries
    top_items = sorted_items[:n]
    worst_items = sorted_items[-n:]

    # Extract country names and values for top n and worst n
    top_countries = [item[0] for item in top_items]
    top_values = [item[1] for item in top_items]

    worst_countries = [item[0] for item in worst_items]
    worst_values = [item[1] for item in worst_items]

    # Conditional plot creation
    if results == 'both':
        # Create subplots for both top and worst n countries
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Top {n} Countries by ' + type, f'Worst {n} Countries by ' + type))

        # Add bar chart for top n countries
        fig.add_trace(go.Bar(
            x=top_countries,
            y=top_values,
            marker_color=color,
            name=f'Top {n}'
        ), row=1, col=1)

        # Add bar chart for worst n countries
        fig.add_trace(go.Bar(
            x=worst_countries,
            y=worst_values,
            marker_color='lightcoral',
            name=f'Worst {n}'
        ), row=1, col=2)

    elif results == 'top':
        # Create a single plot for top n countries
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_countries,
            y=top_values,
            marker_color=color,
            name=f'Top {n}'
        ))
        fig.update_layout(title=f'Top {n} Countries by {type}')

    elif results == 'worst':
        # Create a single plot for worst n countries
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=worst_countries,
            y=worst_values,
            marker_color='lightcoral',
            name=f'Worst {n}'
        ))
        fig.update_layout(title=f'Worst {n} Countries by {type}')

    # Update common layout properties
    fig.update_layout(
        xaxis=dict(tickangle=270, tickfont=dict(size=10)),
        yaxis_title=type,
        height=500
    )

    # Customize individual x-axes for better readability (works only for 'both')
    if results == 'both':
        fig.update_xaxes(title_text="Country", tickangle=270, tickfont=dict(size=10), row=1, col=1)
        fig.update_xaxes(title_text="Country", tickangle=270, tickfont=dict(size=10), row=1, col=2)
    else:
        fig.update_xaxes(title_text="Country", tickangle=270, tickfont=dict(size=10))

    fig.show()

    # Return dictionaries based on the `results` parameter
    if results == 'both':
        return dict(zip(top_countries, top_values)), dict(zip(worst_countries, worst_values))
    elif results == 'top':
        return dict(zip(top_countries, top_values))
    elif results == 'worst':
        return dict(zip(worst_countries, worst_values))
   
def heatmap(measure_dict, type, color='Viridis'):
    countries = list(measure_dict.keys())
    centrality_values = list(measure_dict.values())

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

def plot_attribute(attribute, node_attributes, top_countries, worst_countries):
    """
    Plots the count of government types for top and worst countries and prints the list of countries with their government types.

    Parameters:
    - node_attributes (dict): A dictionary containing attributes for each country, including government type.
    - top_countries (list): A list of top countries to be analyzed.
    - worst_countries (list): A list of worst countries to be analyzed.

    Returns:
    - None: The function will display a bar plot and print government types for the top and worst countries.
    """
    top_country_attribute_values = [node_attributes[country][attribute] for country in top_countries]
    worst_country_attribute_values = [node_attributes[country][attribute] for country in worst_countries]

    top_attribute_counts = Counter(top_country_attribute_values)
    worst_attribute_counts = Counter(worst_country_attribute_values)

    gov_types = set(top_attribute_counts.keys()).union(worst_attribute_counts.keys())
    top_counts = [top_attribute_counts.get(gov_type, 0) for gov_type in gov_types]
    worst_counts = [worst_attribute_counts.get(gov_type, 0) for gov_type in gov_types]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(gov_types),
        y=top_counts,
        name='Top Countries',
        marker_color='skyblue'
    ))

    fig.add_trace(go.Bar(
        x=list(gov_types),
        y=worst_counts,
        name='Worst Countries',
        marker_color='lightcoral'
    ))

    fig.update_layout(
        title=f'Count of {attribute} in Top and Worst Countries',
        xaxis_title=attribute,
        yaxis_title='Count',
        barmode='group'
    )
    fig.show()

    print("Top Countries")
    print(f"\t{'Country':<25} | {attribute:<40}")
    print('\t' + '-' * 65)
    for country in top_countries:
        gov_type = node_attributes[country][attribute]
        print(f"\t{country:<25} | {gov_type:<40}")
    print()

    # Print government types for worst countries
    print("Worst Countries")
    print(f"\t{'Country':<25} | {attribute:<40}")
    print('\t' + '-' * 65)
    for country in worst_countries:
        gov_type = node_attributes[country][attribute]
        print(f"\t{country:<25} | {gov_type:<40}")

def top_centrality_diff(cent1, cent2, type1, type2, typello, top_n=15, color1='skyblue', color2='orange', diff_color='green', ratio_color='purple', relative=False, text=False):
    
    # Calculate the difference and the difference ratio between the two centrality dictionaries
    diff = {country: abs(cent1.get(country, 0) - cent2.get(country, 0)) for country in set(cent1) | set(cent2)}
    ratio = {country: abs(cent1.get(country, 0) - cent2.get(country, 1)) / max(cent1.get(country, 0), cent2.get(country, 1)) for country in set(cent1) | set(cent2)}

    # Sort the countries based on the relative parameter
    if relative:    
        # Sort by relative difference
        top_diff_items = sorted(ratio.items(), key=lambda item: item[1], reverse=True)[:top_n]
    else:
        # Sort by absolute difference
        top_diff_items = sorted(diff.items(), key=lambda item: item[1], reverse=True)[:top_n]

    # Prepare data for the plots
    sorted_countries = [item[0] for item in top_diff_items]  # Get the country names (no need to reverse the list)
    cent1_values = [round(cent1.get(country, 0), 3) for country in sorted_countries]
    cent2_values = [round(cent2.get(country, 0), 3) for country in sorted_countries]
    diff_values = [round(diff.get(country, 0), 3) for country in sorted_countries]  # Get the absolute differences
    ratio_values = [round(ratio.get(country, 0), 3) for country in sorted_countries]  # Get the relative differences

    
    # Create the first bar trace for cent1
    trace1 = go.Bar(
        x=sorted_countries, 
        y=cent1_values, 
        marker_color=color1,
        name=type1,
        text=cent1_values if text else None,
        textposition='auto',
        textfont=dict(size=14)
    )

    # Create the second bar trace for cent2
    trace2 = go.Bar(
        x=sorted_countries,  
        y=cent2_values, 
        marker_color=color2,
        name=type2,
        text=cent2_values if text else None, 
        textposition='auto',
        textfont=dict(size=14)
    )

    # Create the third bar trace for the absolute difference values
    trace3 = go.Bar(
        x=sorted_countries,  
        y=diff_values, 
        marker_color=diff_color,
        name='Absolute Difference',
        text=diff_values if text else None,  
        textposition='auto',
        textfont=dict(size=14)
    )

    # Create the fourth bar trace for the relative difference values
    trace4 = go.Bar(
        x=sorted_countries,  
        y=ratio_values, 
        marker_color=ratio_color,
        name='Relative Difference',
        text=ratio_values if text else None,  
        textposition='auto',
        textfont=dict(size=14)
    )

    # Combine the traces into a single figure
    fig = go.Figure(data=[trace1, trace2, trace3, trace4])

    # Update the layout of the figure
    fig.update_layout(
        title=f'Top {top_n} Countries by {"relative" if relative else "absolute"} difference in {typello}',
        yaxis_title='Centrality Value',
        xaxis_title='Country',
        xaxis=dict(tickfont=dict(size=10)),  
        width=800,  # Adjust width to accommodate four bars
        height=400,  
        barmode='group'  
    )

    # Display the figure
    fig.show()

def simrank_plot(simrank_dict, type):
    simrank_split = {}
    for (country, us_sim), ch_sim in zip(simrank_dict['United States'].items(), simrank_dict['China'].values()):
        if ch_sim > us_sim:
            simrank_split[country] = 'China'
        else:
            simrank_split[country] = 'United States'
    
    countries_us = [country for country, label in simrank_split.items() if label == 'United States']
    countries_china = [country for country, label in simrank_split.items() if label == 'China']

    color_map = {'United States': 'skyblue', 'China': 'lightcoral'}

    fig = go.Figure()

    # United States trace
    fig.add_trace(go.Choropleth(
        locations=countries_us,
        locationmode='country names',
        z=[0] * len(countries_us),  # All 0 since they're all "United States"
        colorscale=[[0, color_map['United States']], [1, color_map['United States']]],
        zmin=0,
        zmax=1,
        showscale=False,
        marker_line_color='black',
        marker_line_width=0.5,
        name='United States',
        hoverinfo='location'
    ))

    # China trace
    fig.add_trace(go.Choropleth(
        locations=countries_china,
        locationmode='country names',
        z=[1] * len(countries_china),  # All 1 since they're all "China"
        colorscale=[[0, color_map['China']], [1, color_map['China']]],
        zmin=0,
        zmax=1,
        showscale=False,
        marker_line_color='black',
        marker_line_width=0.5,
        name='China',
        hoverinfo='location'
    ))

    # Adding a manual legend
    fig.add_trace(go.Scattergeo(
        locationmode='country names',
        locations=[None],  # Empty trace to use for legend
        marker=dict(size=0.1, color=color_map['United States']),
        mode='markers',
        showlegend=True,
        name='United States'
    ))

    fig.add_trace(go.Scattergeo(
        locationmode='country names',
        locations=[None],  # Empty trace to use for legend
        marker=dict(size=0.1, color=color_map['China']),
        mode='markers',
        showlegend=True,
        name='China'
    ))

    fig.update_layout(
        title=f'{type}',
        geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
        height=500,
        width=700,
        legend=dict(
            traceorder="normal",
            itemsizing="constant"
        )
    )

    fig.show()

def plot_centrality_power_law(measure_dict, bins, type, color):
    measure_values = list(measure_dict.values())

    # Calculate histogram
    freq, bins = np.histogram(measure_values, bins=bins)

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

def cumulative_distribution(centrality, type, color):
    # Sort the values
    centrality_values = list(centrality.values())
    sorted_centrality_values = np.sort(centrality_values)[::-1]

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

def extract_node_attributes(graph, attribute):
    return {node: data.get(attribute, 0) for node, data in graph.nodes(data=True)}

def correlation_centrality_gdp(centrality_values, gdp_values, color='skyblue'):
    # Convert the input dictionaries to lists of values
    centrality_values_list = list(centrality_values.values())
    gdp_values_list = list(gdp_values.values())

    # Calculate Pearson correlation
    correlation, p_value = stats.pearsonr(centrality_values_list, gdp_values_list)

    print(f"Pearson correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4e}")

    # Prepare data for Plotly
    data = {
        'PageRank': centrality_values_list,
        'GDP': gdp_values_list
    }

    # Plotting the correlation
    fig = px.scatter(
        data,
        x='PageRank',
        y='GDP',
        title=f'Correlation between PageRank values and GDP\t (Pearson r = {correlation:.4f})',
        labels={'x': 'Centrality', 'y': 'GDP'},
        color_discrete_sequence=[color]
    )

    fig.update_layout(height=500)

    fig.show()
    return correlation, p_value

def count_commodities(node_attributes1, node_attributes2, n=15):
    def get_top_commodities(node_attributes):
        # Initialize a counter to count occurrences of each commodity
        commodity_counter = Counter()

        for country, attributes in node_attributes.items():
            export_commodities = attributes.get('Export Commodities SITC', [])
            commodity_counter.update(export_commodities)

        # Sort commodities by their counts in descending order
        sorted_commodities = sorted(commodity_counter.items(), key=lambda item: item[1], reverse=True)
        
        # Take only the top n commodities
        top_commodities = sorted_commodities[:n]
        return top_commodities

    # Get top commodities for both dictionaries
    top_commodities1 = get_top_commodities(node_attributes1)
    top_commodities2 = get_top_commodities(node_attributes2)

    # Unzipping into two lists for commodities and their counts
    commodities1, counts1 = zip(*top_commodities1)
    commodities2, counts2 = zip(*top_commodities2)

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=("12-core", "Whole Network"))

    # Add bar charts for each dataset
    fig.add_trace(go.Bar(x=commodities1, y=counts1, name="12-core", marker=dict(color='skyblue')), row=1, col=1)
    fig.add_trace(go.Bar(x=commodities2, y=counts2, name="Whole Network", marker=dict(color='blue')), row=1, col=2)

    # Update layout
    fig.update_layout(
        title=f'Top {n} Export Commodities Occurrences',
        xaxis_title="Commodity",
        yaxis_title="Occurrences",
        height=500,
        showlegend=False  # Hide legend to avoid repetition
    )

    fig.update_xaxes(tickfont=dict(size=7), row=1, col=1)
    fig.update_xaxes(tickfont=dict(size=7), row=1, col=2)

    # Display the figure with subplots
    fig.show()
    
    # Return the top n commodities as dictionaries for both datasets
    return dict(top_commodities1), dict(top_commodities2)

def plot_commodity_on_world_map(G, centrality, commodity, title='Commodity Sharing Network Visualization', edges_color='skyblue', highlight_edges_color='lightcoral', markers_color = '#3366cc', highlight_markers_color = '#dc3912'):
    nodes_to_highlight = set()
    edges_to_highlight = set()

    for country, data in G.nodes(data=True):
        export_commodities = data.get('Export Commodities SITC', [])
        commodities = set(export_commodities)
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
        line=dict(width=1, color=edges_color),
        hoverinfo='none'
    )

    highlighted_edge_trace = go.Scattergeo(
        locationmode='ISO-3',
        lon=[],
        lat=[],
        mode='lines',
        line=dict(width=2, color=highlight_edges_color),
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
            color=[highlight_markers_color if node in nodes_to_highlight else markers_color for node in G.nodes() if 'x' in G.nodes[node] and 'y' in G.nodes[node]],
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

# COSE NON USATE (O ANCORA NON USATE)

# Plot the Degree Correlation
# def deg_cor(G, color='skyblue'):
#     # Compute the degree of each node
#     degree = dict(G.degree())

#     # Compute the average neighbor degree for each node
#     avg_neighbor_degree = nx.average_neighbor_degree(G)

#     # Prepare data for plotting
#     degrees = list(degree.values())
#     avg_neighbor_degrees = [avg_neighbor_degree[node] for node in G.nodes()]

#     # Scatter plot
#     fig = px.scatter(
#         x=degrees,
#         y=avg_neighbor_degrees,
#         labels={'x': 'Node Degree', 'y': 'Average Neighbor Degree'},
#         title='Degree Correlation',
#         color_discrete_sequence=[color]
#     )

#     fig.update_layout(height=500)

#     fig.show()

# def extract_node_attributes(graph, attribute):
#     return {node: data.get(attribute, 0) for node, data in graph.nodes(data=True)}

# def find_countries_with_common_commodities(G):
#     commodity_to_countries = defaultdict(list)
#     commodity_to_countries_export = defaultdict(list)
#     commodity_to_countries_import = defaultdict(list)
    
#     for country, data in G.nodes(data=True):
#         export_commodities = data.get('Export Commodities', [])
#         import_commodities = data.get('Import Commodities', [])
#         # Use a set to avoid duplicates
#         commodities = set(export_commodities) | set(import_commodities)
        
#         for commodity in commodities:
#             commodity_to_countries[commodity].append(country)
#         for commodity in export_commodities:
#             commodity_to_countries_export[commodity].append(country)
#         for commodity in import_commodities:
#             commodity_to_countries_import[commodity].append(country)
    
#     return commodity_to_countries, commodity_to_countries_export, commodity_to_countries_import

# def analyze_commodities(node_attributes):
#     commodity_counter_export = Counter()
#     commodity_counter_import = Counter()
#     for country, attributes in node_attributes.items():
#         commodities_export = attributes.get('Export Commodities', [])
#         commodity_counter_export.update(commodities_export)
#         commodities_import = attributes.get('Import Commodities', [])
#         commodity_counter_import.update(commodities_import)
    
#     return commodity_counter_export, commodity_counter_import

# def plot_attributes(data_dict, title, xlabel, ylabel, color='skyblue', colorscale='Viridis'):
#     # Sort the data by country names (keys)
#     sorted_items = sorted(data_dict.items())
#     sorted_countries = [item[0] for item in sorted_items]
#     sorted_values = [item[1] for item in sorted_items]

#     # Create and display the bar chart
#     bar_fig = go.Figure(data=[go.Bar(
#         x=sorted_countries,
#         y=sorted_values,
#         marker_color=color
#     )])

#     bar_fig.update_layout(
#         title=title,
#         xaxis_title=xlabel,
#         yaxis_title=ylabel,
#         xaxis=dict(tickangle=270, tickfont=dict(size=3)),
#         height=500
#     )

#     bar_fig.show()

#     # Create and display the world heatmap
#     heatmap_fig = go.Figure(data=go.Choropleth(
#         locations=sorted_countries,
#         locationmode='country names',
#         z=sorted_values,
#         colorscale=colorscale,
#         colorbar_title=ylabel
#     ))

#     heatmap_fig.update_layout(
#         title=title + ' World Heatmap',
#         height=500
#     )

#     heatmap_fig.show()

# def plot_common_commodities_histogram(commodity_to_countries, name='Network', color='skyblue'):
#     commodity_counts = {commodity: len(countries) for commodity, countries in commodity_to_countries.items()}
#     sorted_items = sorted(commodity_counts.items(), key=lambda item: item[1], reverse=True)
#     commodities = [item[0] for item in sorted_items]
#     counts = [item[1] for item in sorted_items]

#     fig = go.Figure(data=[go.Bar(
#         x=commodities,
#         y=counts,
#         marker_color=color
#     )])

#     fig.update_layout(
#         title=f'Number of Countries Sharing Each Commodity in the {name}',
#         xaxis_title='Commodity',
#         yaxis_title='Number of Countries',
#         xaxis=dict(tickangle=270, tickfont=dict(size=2)),
#         height=500
#     )

#     fig.show()

# def print_common_commodities(commodity_to_countries, type='Export'):
#     for commodity, countries in commodity_to_countries.items():
#         if len(countries) > 1:
#             print(f"Commodity '{commodity}' is shared by countries: {', '.join(countries)}")

# def avg_shortestpath(G):
#     if nx.is_strongly_connected(G):
#         avg_shortest_path_length = nx.average_shortest_path_length(G)
#         print(f"Average Shortest Path Length: {avg_shortest_path_length:.4f}")
#     else:
#         # For weakly connected graphs, compute the average for each strongly connected component
#         components = list(nx.strongly_connected_components(G))
#         total_length = 0
#         total_nodes = 0
        
#         for component in components:
#             if len(component) > 1:
#                 subgraph = G.subgraph(component)
#                 avg_length = nx.average_shortest_path_length(subgraph)
#                 total_length += avg_length * len(component)
#                 total_nodes += len(component)
        
#         avg_shortest_path_length = total_length / total_nodes if total_nodes > 0 else float('inf')
#         print(f"Average Shortest Path Length (average of strongly connected components): {avg_shortest_path_length:.4f}")
#     return avg_shortest_path_length


