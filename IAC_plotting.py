# PLOTTING FUNCTIONS FOR USE IN JUPYTER NOTEBOOKS

import numpy as np
import pandas as pd
import panel as pn
import networkx as nx
import plotly.graph_objs as go
import plotly.express as px

import ipywidgets as widgets
from IPython.display import display

import socket

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

import asyncio
import threading

from itertools import combinations
import subprocess

# node color palette
col_pal = px.colors.qualitative.Plotly

def test():
    return

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
            
# changing node size based on activation (sum of connection weights)
# Default activation is -0.1
def get_activation(graph, rest_act=-0.1):
    activations = {node: rest_act for node in graph.nodes()}
    for u,v,d in graph.edges(data=True):
        activations[u] += d["weight"]
        if u != v:
            activations[v] += d["weight"]
    return activations, rest_act

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
            hidden_connections = df[df[c]==n.split(': ')[1]][hidden_state].unique()
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

# Function to create the plot
def create_plot(pos, pools, G, hover_node=None):
    # creating figure
    fig = go.Figure()
    
    # getting node sizes, node colors, connection linewidths
    activations, rest_act = get_activation(G)
    sizes = activation_to_size(activations)
    colors = [col_pal[list(set(pools.values())).index(pools[n])] for n in G.nodes()]
    linewidths = connection_to_lw(G.edges(data=True))
    
    # setting up node outline colors
    outline_colors = colors.copy()
    
    if hover_node is not None:
        hidden_ind = list(G.nodes()).index(hover_node)
        for edge_group, line_width in zip(G.edges(data=True), linewidths):
            u,v,d=edge_group
            x, y = [], []
            if hover_node in {u,v}:
                x += [pos[u][0], pos[v][0], None]
                y += [pos[u][1], pos[v][1], None]
            
                # connection line color
                line_color = 'blue' if pools[u] != pools[v] else 'red'
            
                # node outline colors
                adjust_ind = list(G.nodes()).index(u) if hover_node != u else list(G.nodes()).index(v)
                outline_colors[adjust_ind] = line_color
                outline_colors[hidden_ind] = 'black' # selected node line color
                colors[hidden_ind] = 'yellow' # selected node fill color
            
                fig.add_trace(go.Scatter(
                            x=x, y=y,
                            line=dict(width=line_width, color=line_color),
                            hoverinfo='none',
                            mode='lines',
                            showlegend=False,)
                          )
    
    # Adding Nodes
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        text=[f"{n.split(': ')[1]}" for n in G.nodes()],
        marker=dict(size=sizes, 
                    color=colors,
                    line=dict(width=2, color=outline_colors)
                   ),
        hoverinfo="text",
        hovertext=[f"Activation: {activations[n]}<br>Net Input: {activations[n]-rest_act}" for n in G.nodes()],
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
    for i in range(len(list(set(pools.values())))):
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
        width=700,
        height=700,
        plot_bgcolor="white",
        title = "Interactive Activation and Competition Network"
    )

    return fig

def plot(df, hidden_state=None):
    '''
    Takes in DataFrame and creates visualization
    '''
    pos, pools, G = init_graph(df, hidden_state)
    pos_map = {tuple(v):k for k,v in pos.items()}
    
    # setting up Dash app
    app = dash.Dash(__name__)

    # App layout
    app.layout = html.Div([
        dcc.Graph(
            id = 'network-graph', # figure
            figure = create_plot(pos, pools, G),
            style = {'margin-top': '50px'}
        ),
        html.Button( # reset button (Dash not compatible with ipywidgets, need html Button instead)
            id = 'reset-button',
            children = 'Reset Network',
            style = {
                'position': 'absolute',
                'top': '20px',
                'left': '20px',
                'background-color': 'skyblue',
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'padding': '5px 5px',
                'cursor': 'pointer'
            }
        ),
        dcc.Store(
            id = 'selected-nodes', # place to store selected nodes
            data = []
        ),
    ])

    @app.callback(
         [Output('network-graph', 'figure'),   # Output for figure
          Output('selected-nodes', 'data')],   # Output for node storage
         [Input('network-graph', 'hoverData'), # Input for hover event
          Input('network-graph', 'clickData'), # Input for click event
          Input('reset-button', 'n_clicks')],  # Input for reset button click
         [State('selected-nodes', 'data'),     # State of selected-nodes
          State('network-graph', 'figure')],   # State of figure
    )
    def fig_update(hoverData, clickData, n_clicks, data, figure):
        '''
        returns: fig, clicked_nodes
        '''
        fig = go.Figure(figure)
        event = dash.callback_context
        trig_event = event.triggered[0]['prop_id']
    
        if 'network-graph' in trig_event:
            # Identify selected node (using location in case node 'text' is not unique)
            selected_node = pos_map[(hoverData['points'][0]['x'], hoverData['points'][0]['y'])]
            node_ind = list(G.nodes()).index(selected_node)
    
        if trig_event == 'network-graph.hoverData' and hoverData and 'points' in hoverData:
            fig = create_plot(pos, pools, G, hover_node=selected_node)
    
        if trig_event == 'network-graph.clickData' and clickData and 'points' in clickData:
            # toggle back to normal if already selected
            if selected_node in data:
                data.remove(selected_node)
            
                # resetting color back to original color
                orig_colors = [col_pal[list(set(pools.values())).index(pools[n])] for n in G.nodes()]
                for trace in fig.data:
                    if 'marker' in trace and 'markers+text' in trace.mode:
                        curr_colors = list(trace.marker.color)
                        curr_colors[node_ind] = 'yellow' #orig_colors[node_ind]
                        trace.marker.color = curr_colors
                    
            # else add node to selected-nodes
            else:
                data.append(selected_node)
    
        if trig_event == 'reset-button.n_clicks':
            fig = create_plot(pos, pools, G)
            data = []
        
        for clicked_node in data:
            clicked_ind = list(G.nodes()).index(clicked_node)
        
            for trace in fig.data:
                if 'marker' in trace and 'markers+text' in trace.mode:
                    curr_colors = list(trace.marker.color)
                    curr_colors[clicked_ind] = 'lightcyan'
                    trace.marker.color = curr_colors
    
        return fig, data
    
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
        port = '8050'
        free_local_port()
    
    app.run(port=port)