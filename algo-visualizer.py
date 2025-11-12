import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Dict, Tuple
import time

st.set_page_config(page_title="Algorithm Visualizer", layout="wide")

# ============================================================================
# ALGORITHM IMPLEMENTATIONS
# ============================================================================

class GraphVisualizer:
    """Base class for graph algorithms with visualization support"""
    
    def __init__(self, graph: nx.Graph, start_node: int):
        self.graph = graph
        self.start_node = start_node
        self.visited = []
        self.order = []
        self.edges_visited = []
        
    def get_steps(self) -> List[Dict]:
        """Returns list of steps for visualization. Override in subclasses."""
        raise NotImplementedError
        
    def visualize_step(self, step: Dict, ax):
        """Draw the graph with current step state"""
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw all edges
        nx.draw_networkx_edges(self.graph, pos, ax=ax, width=2, alpha=0.3, edge_color='gray')
        
        # Draw visited edges in red
        if step.get('edges_visited'):
            nx.draw_networkx_edges(
                self.graph, pos, 
                edgelist=step['edges_visited'],
                ax=ax, width=3, edge_color='red', alpha=0.7
            )
        
        # Color nodes based on state
        node_colors = []
        for node in self.graph.nodes():
            if node == step.get('current'):
                node_colors.append('gold')  # Current node
            elif node in step.get('visited', []):
                node_colors.append('lightgreen')  # Visited
            else:
                node_colors.append('lightblue')  # Unvisited
        
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                              node_size=800, ax=ax)
        nx.draw_networkx_labels(self.graph, pos, font_size=12, 
                               font_weight='bold', ax=ax)
        
        ax.set_title(f"{step['title']}", fontsize=14, fontweight='bold')
        ax.axis('off')


class BFSVisualizer(GraphVisualizer):
    """Breadth-First Search visualization"""
    
    def get_steps(self) -> List[Dict]:
        steps = []
        visited = set()
        queue = deque([self.start_node])
        visited.add(self.start_node)
        edges_visited = []
        
        steps.append({
            'title': f'Start: Add {self.start_node} to queue',
            'visited': list(visited),
            'current': self.start_node,
            'edges_visited': []
        })
        
        while queue:
            u = queue.popleft()
            
            steps.append({
                'title': f'Process node {u} from queue',
                'visited': list(visited),
                'current': u,
                'edges_visited': edges_visited
            })
            
            for neighbor in sorted(self.graph.neighbors(u)):
                edge = (u, neighbor) if u < neighbor else (neighbor, u)
                
                # Check neighbor
                if neighbor not in visited:
                    # Tree edge: neighbor is unvisited
                    edges_visited.append(edge)
                    steps.append({
                        'title': f'Tree edge {u}→{neighbor} (unvisited). Queue: push {neighbor}',
                        'visited': list(visited),
                        'current': u,
                        'edges_visited': edges_visited
                    })
                    visited.add(neighbor)
                    queue.append(neighbor)
                    steps.append({
                        'title': f'Mark {neighbor} as visited',
                        'visited': list(visited),
                        'current': u,
                        'edges_visited': edges_visited
                    })
                else:
                    # Non-tree edge: neighbor already visited, ignore
                    steps.append({
                        'title': f'Edge {u}→{neighbor} (already visited). Ignore.',
                        'visited': list(visited),
                        'current': u,
                        'edges_visited': edges_visited
                    })
        
        steps.append({
            'title': f'BFS Complete! Visited: {sorted(list(visited))}',
            'visited': list(visited),
            'current': None,
            'edges_visited': edges_visited
        })
        
        return steps


class DFSVisualizer(GraphVisualizer):
    """Depth-First Search visualization"""
    
    def get_steps(self) -> List[Dict]:
        steps = []
        visited = set()
        stack = [self.start_node]
        edges_visited = []
        
        steps.append({
            'title': f'Start: Push {self.start_node} to stack',
            'visited': list(visited),
            'current': self.start_node,
            'edges_visited': []
        })
        
        while stack:
            u = stack.pop()
            
            if u in visited:
                steps.append({
                    'title': f'Pop {u} from stack (already visited). Skip.',
                    'visited': list(visited),
                    'current': None,
                    'edges_visited': edges_visited
                })
                continue
            
            visited.add(u)
            steps.append({
                'title': f'Pop {u} from stack. Mark as visited.',
                'visited': list(visited),
                'current': u,
                'edges_visited': edges_visited
            })
            
            neighbors = sorted(self.graph.neighbors(u), reverse=True)
            
            for neighbor in neighbors:
                edge = (u, neighbor) if u < neighbor else (neighbor, u)
                
                if neighbor not in visited:
                    # Tree edge: push unvisited neighbor
                    edges_visited.append(edge)
                    stack.append(neighbor)
                    steps.append({
                        'title': f'Tree edge {u}→{neighbor} (unvisited). Push {neighbor} to stack.',
                        'visited': list(visited),
                        'current': u,
                        'edges_visited': edges_visited
                    })
                else:
                    # Non-tree edge: neighbor already visited, ignore
                    steps.append({
                        'title': f'Edge {u}→{neighbor} (already visited). Ignore.',
                        'visited': list(visited),
                        'current': u,
                        'edges_visited': edges_visited
                    })
        
        steps.append({
            'title': f'DFS Complete! Visited: {sorted(list(visited))}',
            'visited': list(visited),
            'current': None,
            'edges_visited': edges_visited
        })
        
        return steps


# Algorithm registry for easy scaling
ALGORITHMS = {
    'BFS': BFSVisualizer,
    'DFS': DFSVisualizer,
}


def create_sample_graph():
    """Create a sample graph for visualization"""
    G = nx.Graph()
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]
    G.add_edges_from(edges)
    return G


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🎨 Algorithm Visualizer")
st.markdown("Visualize graph traversal algorithms step by step")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Controls")
    
    algorithm = st.selectbox(
        "Select Algorithm",
        options=list(ALGORITHMS.keys()),
        help="Choose which algorithm to visualize"
    )
    
    start_node = st.number_input(
        "Start Node",
        min_value=0,
        max_value=5,
        value=0,
        help="Node to start traversal from"
    )
    
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.current_step = 0

with col2:
    st.subheader("📊 Visualization")
    
    # Create graph
    G = create_sample_graph()
    
    # Validate start node
    if start_node not in G.nodes():
        st.error(f"Start node {start_node} not in graph!")
        st.stop()
    
    # Get algorithm steps
    visualizer = ALGORITHMS[algorithm](G, start_node)
    steps = visualizer.get_steps()
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    
    # Display current step
    fig, ax = plt.subplots(figsize=(8, 6))
    current_step_data = steps[st.session_state.current_step]
    visualizer.visualize_step(current_step_data, ax)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("📍 Step Navigation")
    
    step_slider = st.slider(
        "Go to step",
        min_value=0,
        max_value=len(steps) - 1,
        value=st.session_state.current_step,
        label_visibility="collapsed"
    )
    if step_slider != st.session_state.current_step:
        st.session_state.current_step = step_slider
        st.rerun()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ Previous Step"):
            st.session_state.current_step = max(0, st.session_state.current_step - 1)
            st.rerun()
    
    with col2:
        st.metric("Step", f"{st.session_state.current_step + 1}/{len(steps)}")
    
    with col3:
        if st.button("Next Step ➡️"):
            st.session_state.current_step = min(len(steps) - 1, st.session_state.current_step + 1)
            st.rerun()

# Legend
st.markdown("---")
st.subheader("📋 Legend")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🔵 **Unvisited** - Not yet explored")
with col2:
    st.markdown("🟢 **Visited** - Already explored")
with col3:
    st.markdown("🟡 **Current** - Currently processing")