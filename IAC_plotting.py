# PLOTTING FUNCTIONS FOR USE IN JUPYTER NOTEBOOKS
# plot() takes in pandas DataFrame and a column name (hidden unit column) and produces an interactive Dash app

import numpy as np
import pandas as pd
import panel as pn
import networkx as nx
import plotly.graph_objs as go
import plotly.express as px
import json
import random

import ipywidgets as widgets
from IPython.display import display

import socket

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

from itertools import combinations
import subprocess

# ---------- IAC MODEL PARAMETERS ----------

MAX = 1.0    # maximum activation
MIN = -0.2   # minimum activation
REST = -0.1  # resting activation
DECAY = 0.1  # decay rate
ESTR = 0.4   # external input weight
ALPHA = 0.1  # internal excitatory input weight
GAMMA = 0.1  # internal inhibitory input weight

# node color palette
col_pal = px.colors.qualitative.Plotly

# ---------- HOSTING FUNCTIONS ----------
def free_local_port(port = '8050'):
    # freeing local port if necessary (to run Dash app)
    port_process = subprocess.run(["lsof", "-i", f":{port}"], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, 
                                  text=True)
    if len(port_process.stdout) > 1:
        text = port_process.stdout
        print(text)
        info = {}
        lines = [l.split() for l in text.split('\n')]
        for i in range(len(lines[0])):
            field, data = lines[0][i], lines[1][i]
            info[field] = data
        
        if info.get('PID', None) is not None:
            result = subprocess.run(["kill", "-9", info['PID']])
            
def find_free_port():
    with socket.socket() as s:
        s.bind(('', 0))            # Bind to a free port provided by the host.
        return s.getsockname()[1]

    
# ---------- NODE/LINE PROPERTY FUNCTIONS ----------

# Map activations to marker sizes (e.g. between 20 and 50)
# Makes all nodes 35 if equal
def activation_to_size(act, low_bound=20, high_bound=50):
    max_act = max(act.values())
    min_act = min(act.values())
    if max_act == min_act:
        return (low_bound + high_bound)//2
    dynam_sizes = [low_bound + (x-min_act) * (high_bound-low_bound)/(max_act-min_act)
                   for x in act.values()]
    return dynam_sizes

# Map activations to node fill colors
# keeps the default fill color if the activation is within (REST-0.05, REST+0.05)
def activation_to_color(act, default_colors, 
                        maxact=MAX, 
                        minact=MIN, 
                        restact=REST):
    colors = []
    bins = []
    # if we have inhibitory activations
    if minact < restact:
        bins += list(np.linspace(minact, restact-0.1, 10))
        colors += ['#9C191B', '#AC1C1E', '#BD1F21', '#D02224', '#DD2C2F',
                   '#E35053', '#E66063', '#EC8385', '#F1A7A9', '#F6CACC']
    # if we have excitatory activations
    if maxact > restact:
        bins += list(np.linspace(restact, maxact, 10))
        colors += ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5',
                   '#2196F3', '#1E88E5', '#1976D2', '#1565C0', '#0D47A1']
        
    if len(bins) != 0:
        # digitize returns indices corresponding to the bins each value falls into
        bin_ind = np.digitize(list(act.values()), bins) - 1
    
    node_colors = []
    for i in range(len(act.values())):
        if np.abs(list(act.values())[i] - restact) > 0.05: 
            # condition only occurs if len(bin_ind) > 0
            node_colors.append(colors[bin_ind[i]])
        else:
            node_colors.append(default_colors[i])
            
    return node_colors

# Map connection strengths to linewidths (e.g. between 1 and 5)
# Makes all nodes 2 if equal
def connection_to_lw(edges, low_bound=1, high_bound=5):
    connections = [np.abs(d['weight']) for u,v,d in edges]
    min_con = min(connections)
    max_con = max(connections)
    if min_con == max_con:
        return [(low_bound + high_bound)//2]*len(edges)
    dynam_lw = [low_bound + (x-min_con) * (high_bound-low_bound)/(max_con-min_con) 
                for x in connections]
    return dynam_lw

def position_nodes(pools):
    # Reference: Axel Cleeremans
    # place nodes in concentric circles with breaks between pools
    
    pos = {}
    nodes = list(pools.keys())
    pool_ids = list(pools.values())
    pool_lens = {str(p):int(c) for p,c in zip(*np.unique(pool_ids, return_counts=True))}
    
    interval = 800 # distance between nodes in each ring
    center_coord = (0,0) # center of all rings
    radius = 1000 # radius of starting ring (incremented by 1000 each circle)
    
    i = len(nodes)-1 # start from hidden units
    prev_pool = pool_ids[-1]
    while i >= 0:
        circum = 2 * np.pi * radius
        # number of nodes per ring
        ring_size = int(circum / interval)
        
        for j in range(ring_size):
            if i < 0:
                break
            
            # changing pools
            curr_pool = pool_ids[i]
            skip = (curr_pool != prev_pool)
            prev_pool = curr_pool
            
            if skip:
                # moving to a new ring if pool won't fit
                if (j > 0) and (pool_lens[curr_pool] > (ring_size)-(j+1)):
                    break
            
                # else staying in current ring and moving a space
                else:
                    continue
            
            # placing node
            angle = 270 + (j * (360/ring_size))
            x = center_coord[0] + (np.cos(np.radians(angle)) * radius)
            y = center_coord[1] + (np.sin(np.radians(angle)) * radius)
            pos[nodes[i]] = [x,y]
            i -= 1
        
        radius += 1000
                 
    return pos

# ---------- GRAPH INITIALIZATION FUNCTION ----------

def init_graph(df, hidden_state = None):
    '''
    Arguments
    - df: DataFrame
          data to plot
    - hidden_state: string
          column to use for hidden nodes
    Returns
    - pos: dict
          mapping from node to position
    - pools: dict
          mapping from node to pool ID
    - G: networkx Graph object
          graph of network (nodes and edges)
    '''
    df = df.copy().dropna()
    assert len(df) > 0, f"DataFrame currently has shape {df.shape}. Must have more than 0 non-null rows."
    
    # converting to type str
    df = df.astype(str)
    
    # default hidden state (first column)
    if hidden_state is None:
        hidden_state = df.columns[0]
    pools = {}
    for c in df.columns:
        nodes = df[c].unique()
        for n in nodes:
            # make unique column/node ID in case columns have identical entries
            pools[f"{c}: {n}"] = c
    for hidden_node in df[hidden_state].unique():
        pools[f"hidden: {hidden_node}"] = 'hidden'
        
    # getting nodes
    nodes = pools.keys()
    
    # getting edges
    edges = []
    # pool nodes only have edges to hidden units and within-pool nodes
    hidden_nodes = [k for k,v in pools.items() if v == 'hidden']
    for c in df.columns:
        pool_nodes = [k for k,v in pools.items() if v == c]
        
        # hidden unit connections
        for n in pool_nodes:
            hidden_connections = df[df[c]==n.split(': ')[-1]][hidden_state].unique()
            edges.extend([(f'hidden: {h}', n, 0) for h in hidden_connections])

        # within-pool connections
        edges.extend([(u, v, 0) for u, v in combinations(pool_nodes, 2)])
        
    # within-pool hidden unit connections
    edges.extend([(u, v, 0) for u, v in combinations(hidden_nodes, 2)])
        
    # creating graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)
    
    # positions
    pos = position_nodes(pools)
        
    return pos, pools, G


# -------- NETWORK UPDATE FUNCTIONS ---------

# GET NET INPUT FOR A NODE
def get_netinput(node, G, pools, unit_info, extinput):
    '''
    Finds the net input for a node
    '''
    inhibitory, excitatory = 0.0, 0.0
    for u, v, w in G.edges(data=True):
        if node not in (u,v):
            continue
        # activation of connected node
        act_neighbor = unit_info['activations'][u] if u != node else unit_info['activations'][v]
        
        # continuing if activation > 0
        if act_neighbor > 0:
            if pools[u] == pools[v]: # inhibitory connection
                inhibitory += -1.0 * act_neighbor
            
            elif pools[u] != pools[v]: # excitatory connection
                excitatory += 1.0 * act_neighbor
    
    net = (GAMMA * inhibitory) + (ALPHA * excitatory) + (ESTR * extinput)
    return net

# UPDATE NODE ACTIVATIONS, NET INPUTS, GRAPH EDGE WEIGHTS
def run_simulation(G, pools, unit_info, clicked_nodes, num_cycles):
    '''
    Updates unit activations according to the IAC update rule
    Arguments:
    - G: networkx Graph object representing network
    - pools: dictionary linking nodes and pools
    - clicked_nodes: list of activated nodes
    - unit_info: 
    - num_cycles: number of cycles to run update
    '''
    G = G.copy()
    
    for i in range(num_cycles):
        
        # updating net inputs
        for node in G.nodes():
            extinput = 1.0 if node in clicked_nodes else 0.0
            unit_info['NetInputs'][node] = get_netinput(node, G, pools, 
                                                        unit_info, extinput)
            
        # updating activations
        for node in G.nodes():
            curr_act = unit_info['activations'][node]
            curr_net = unit_info['NetInputs'][node]
            
            if curr_net > 0:
                delta_act = (MAX - curr_act) * curr_net - DECAY * (curr_act - REST)
            else:
                delta_act = (curr_act - MIN) * curr_net - DECAY * (curr_act - REST)
                
            nxt_act = curr_act + delta_act
            # bounding the activation
            nxt_act = min(MAX, nxt_act)
            nxt_act = max(MIN, nxt_act)
                
            unit_info['activations'][node] = nxt_act
              
    
    # updating weights (making activated nodes' edges non-zero)
    edges = [e for e in G.edges(data=True)]
    for u, v, w in edges:
        # if connection contains activated node, display it
        if unit_info['activations'][u] > 0 or unit_info['activations'][v] > 0:
            new_weight = 1.0 if pools[u] != pools[v] else -1.0
            G.remove_edges_from([(u,v,w)])
            G.add_edge(u, v, weight = new_weight)
            
    return G

# ---------- PLOTTING FUNCTIONS ----------

# Function to create the plot
def create_plot(pos, pools, G, unit_info, hover_node=None):
    # creating figure
    fig = go.Figure()
    
    # getting node sizes, node colors, connection linewidths
    default_colors = [col_pal[list(set(pools.values())).index(pools[n])] for n in G.nodes()]
    colors = activation_to_color(unit_info['activations'], default_colors)
    
    # plotting non-zero connections
    for edge_group in G.edges(data=True):
        u, v, d = edge_group
        # skipping 0 weight
        if d['weight'] == 0:
            continue
        # skipping hover_node connections
        if (hover_node is not None and hover_node in {u,v}):
            continue
        x = [pos[u][0], pos[v][0], None]
        y = [pos[u][1], pos[v][1], None]
        
        # line color (light blue and light red)
        line_color = '#8181ff' if pools[u] != pools[v] else '#ff8181'
        
        fig.add_trace(go.Scatter(
                    x=x, y=y, 
                    line=dict(width=2, color=line_color),
                    hoverinfo='none',
                    mode='lines',
                    showlegend=False)
                  )
    
    # setting up node outline colors
    outline_colors = default_colors.copy()
    
    # plotting hover node if needed
    if hover_node is not None:
        hover_ind = list(G.nodes()).index(hover_node)
        for edge_group in G.edges(data=True):
            u,v,d=edge_group
            x, y = [], []
            if hover_node in {u,v}:
                x += [pos[u][0], pos[v][0], None]
                y += [pos[u][1], pos[v][1], None]
            
                # connection line color
                line_color = 'blue' if pools[u] != pools[v] else 'red'
            
                # node outline and fill colors
                outline_colors[hover_ind] = 'black' # selected node line color
                colors[hover_ind] = 'yellow' # selected node fill color
            
                fig.add_trace(go.Scatter(
                            x=x, y=y,
                            line=dict(width=3, color=line_color),
                            hoverinfo='none',
                            mode='lines',
                            showlegend=False,)
                          )

    # Adding Nodes
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        text=[f"{n.split(': ')[-1]}" for n in G.nodes()],
        marker=dict(size=30, 
                    color=colors,
                    line=dict(width=6, color=outline_colors),
                    opacity=1.0,
                   ),
        hoverinfo="text",
        hovertext=[f"Activation: {unit_info['activations'][n]:.3f}<br>Net Input: {unit_info['NetInputs'][n]:.3f}" for n in G.nodes()],
        mode="markers+text",
        showlegend=False)
    fig.add_trace(node_trace)
    
    # Adding Legend Labels (dummy traces)
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        line=dict(width=2, color="blue"),
        mode="lines",
        name="excitatory",
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        line=dict(width=2, color="red"),
        mode="lines",
        name="inhibitory",
        showlegend=True
    ))
    for i in range(len(set(pools.values()))):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            marker=dict(size=20,color=col_pal[i]),
            mode="markers",
            name=list(set(pools.values()))[i],
            showlegend=True
        ))
    
    # removing grid and axes
    fig.update_layout(
        showlegend=True,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=675,
        height=675,
        plot_bgcolor="white",
        margin=dict(t=30, l=20, r=20),
        title = "Interactive Activation and Competition Network"
    )

    return fig


def serialize_graph(G):
    '''
    Converts Graph to JSON-serialized object for use in JavaScript documents
    '''
    return nx.node_link_data(G, 
                             edges='edges',
                             nodes='nodes')

def deserialize_graph(data):
    '''
    Converts JSON-serialized object back to NetworkX Graph object
    '''
    return nx.node_link_graph(data,
                              edges='edges',
                              nodes='nodes')

def serialize_dict(d):
    '''
    Converts dictionary to JSON-serialized object for use in JavaScript documents
    '''
    return json.dumps(d)

def deserialize_dict(data):
    '''
    Converts JSON-serialized object back to dictionary
    '''
    return json.loads(data)

    
def plot(df, hidden_state=None):
    '''
    Takes in DataFrame and creates visualization
    '''
    # initializing graph
    pos, pools, G = init_graph(df, hidden_state)
    pos_map = {tuple(v):k for k,v in pos.items()}
    
    # setting initial unit info (activation and net input)
    unit_info = {'activations': {node: REST for node in G.nodes()},
                 'NetInputs': {node: 0.0 for node in G.nodes()}}
    
    # setting up Dash app
    app = dash.Dash(__name__)

    # App layout
    app.layout = html.Div(
        style={'display': 'flex', 'height':'90vh'},
        children=[
            # control panel (left)
            html.Div(
                style={
                    'width': '115px',
                    'padding': '10px',
                    'background-color': '#f8f9fa',
                    'box-shadow': '2px 0 5px rgba(0, 0, 0, 0.1)',
                    'flex':'none', # fixed size 
                },
                children=[
                    html.Button( # reset button 
                        id = 'reset-button',
                        children = 'Reset Network',
                        style = {
                            'background-color': '#4fb3db',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '5px',
                            'padding': '5px 5px',
                            'cursor': 'pointer',
                            'margin-bottom': '10px'
                        }
                    ),
                    html.Div( # box for slider output
                        id='slider-output',
                        style={
                            'margin-bottom': '10px',
                            'font-size': '14px',
                            'text-align': 'center',
                        }
                    ),
                    dcc.Slider(0,1000, # slider for simulation
                       value=0,
                       id='num-cycles',
                       vertical=True,
                    ),
                    html.Button( # run simulation button
                        id = 'run-simulation',
                        children = 'Run Simulation',
                        style = {
                            'background-color': '#FA5F55',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '5px',
                            'padding': '5px 5px',
                            'cursor': 'pointer',
                            'margin-top': '10px',
                        }
                    ),
                    dcc.Store(
                        id = 'selected-nodes', # place to store selected nodes
                        data = []
                    ),
                ]
            ),
            # network (right)
            html.Div(
                style={'flex':'1', # will resize based on available space
                       'padding': '10px',
                },
                children=[
                    dcc.Graph(
                        id = 'network-graph', # figure
                        figure = create_plot(pos, pools, G, unit_info),
                    ),
                    dcc.Store(
                        id = 'graph-state', # store graph
                        data = serialize_graph(G)
                    ),
                    dcc.Store(
                        id = 'unit-info', # store activations, net input
                        data = serialize_dict(unit_info)
                    ),
                ]
            )
        ]
    )

    @app.callback(
         [Output('network-graph', 'figure'),   # Output for figure
          Output('graph-state', 'data'),       # Output for graph
          Output('unit-info', 'data'),         # Output for Units
          Output('selected-nodes', 'data'),    # Output for node storage
          Output('slider-output', 'children'), # Output for slider container
          Output('num-cycles', 'value')],      # Output for slider value
         [Input('network-graph', 'hoverData'), # Input for hover event
          Input('network-graph', 'clickData'), # Input for click event
          Input('reset-button', 'n_clicks'),   # Input for reset button click
          Input('run-simulation', 'n_clicks'), # Input for simulation button click
          Input('num-cycles', 'value')],       # Input for slider
         [State('selected-nodes', 'data'),     # State of selected-nodes
          State('network-graph', 'figure'),    # State of figure
          State('graph-state', 'data'),        # State of Graph 
          State('unit-info', 'data')]          # State of Units
    )
    def fig_update(hoverData,
                   clickData,
                   n_clicks_reset, 
                   n_clicks_sim, 
                   value, 
                   selected_data, 
                   figure,
                   graph_data,
                   unit_data):
        '''
        returns: fig, graph state, unit info state, 
                 clicked_nodes, slider_text, slider_value
        '''
        
        G = deserialize_graph(graph_data)
        unit_info = deserialize_dict(unit_data)
        
        fig = go.Figure(figure)
        event = dash.callback_context
        trig_event = event.triggered[0]['prop_id']
    
        if 'network-graph' in trig_event:
            # Identify selected node (using location in case node 'text' is not unique)
            selected_node = pos_map[(hoverData['points'][0]['x'], hoverData['points'][0]['y'])]
            node_ind = list(G.nodes()).index(selected_node)
    
        if trig_event == 'network-graph.hoverData' and hoverData and 'points' in hoverData:
            fig = create_plot(pos, pools, G, unit_info, hover_node=selected_node)
    
        if trig_event == 'network-graph.clickData' and clickData and 'points' in clickData:
            # toggle back to normal if already selected
            if selected_node in selected_data:
                selected_data.remove(selected_node)
            
                # resetting color back to hover color
                for trace in fig.data:
                    if 'marker' in trace and 'markers+text' in trace.mode:
                        curr_colors = list(trace.marker.color)
                        curr_colors[node_ind] = 'yellow'
                        trace.marker.color = curr_colors
                    
            # else add node to selected-nodes
            else:
                selected_data.append(selected_node)
    
        if trig_event == 'reset-button.n_clicks':
            # resetting all Graph connections to 0
            edges = [e for e in G.edges(data=True)]
            for u,v,w in edges:
                G.remove_edges_from([(u,v,w)])
                G.add_edge(u, v, weight=0)
                
            # resetting unit_info
            unit_info['activations'] = {node: REST for node in G.nodes()}
            unit_info['NetInputs'] = {node: 0.0 for node in G.nodes()}
            
            # creating default figure
            fig = create_plot(pos, pools, G, unit_info)
            
            # emptying clicked nodes
            selected_data = []
            
            # resetting slider value
            value = 0
        
        if trig_event == 'run-simulation.n_clicks':
            G = run_simulation(G, pools, unit_info,
                                clicked_nodes = selected_data, 
                                num_cycles = value)
            
            # print statement to help keep track of results
            #print({k: v for k, v in sorted(unit_info['activations'].items(), reverse=True, key=lambda item: item[1])})
            fig = create_plot(pos, pools, G, unit_info)
        
        for clicked_node in selected_data:
            clicked_ind = list(G.nodes()).index(clicked_node)
        
            for trace in fig.data:
                # identifying node trace
                if 'marker' in trace and 'markers+text' in trace.mode:
                    curr_colors = list(trace.marker.color)
                    # updating selected node fill color
                    curr_colors[clicked_ind] = '#31fa01' # lime green
                    trace.marker.color = curr_colors
    
        slider_text = f"Number of Update Cycles: {value}"
        graph_data = serialize_graph(G)
        unit_data = serialize_dict(unit_info)
    
        return fig, graph_data, unit_data, selected_data, slider_text, value
    
    try:
        port = find_free_port()
    except:
        print("No ports are available. The below process is running on local port 8050:\n")
        local_port = subprocess.run(["lsof", "-i", f":8050"], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, 
                                  text=True)
        print(local_port.stdout)
        user_in = input("\nWould you like to kill the above process to make space? [y/n] ")
        if user_in.strip().lower() not in ['y', 'yes']:
            print('Aborting Visualization...')
            return
        print('Freeing port 8050...')
        port = '8050'
        free_local_port()
    
    app.run(port=port)