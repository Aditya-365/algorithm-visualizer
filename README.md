# algorithm-visualizer

An **interactive visualization tool** built with [Streamlit](https://streamlit.io/) for exploring and understanding **graph traversal algorithms** such as **Breadth-First Search (BFS)** and **Depth-First Search (DFS)**.  

The app visually demonstrates how these algorithms traverse a graph **step-by-step**, highlighting visited nodes, traversal order, and the structure of the graph in real time.

---

## Features

- **Interactive algorithm visualization** for BFS & DFS  
- **Step-by-step traversal control** with Previous/Next navigation  
- **Live graph drawing** using NetworkX & Matplotlib  
- **Color-coded states** for unvisited, current, and visited nodes  
- **Session-aware state management** for seamless exploration  
- **Built-in controls** for algorithm selection and start node input  

---

## Tech Stack

| Library | Purpose |
|----------|----------|
| **Streamlit** | Interactive web UI for Python apps |
| **NetworkX** | Graph data structure and traversal logic |
| **Matplotlib** | Graph plotting and visualization |
| **Python 3.10+** | Core runtime environment |

---

## Installation 
### **1. Clone the repository**
```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
```

### **2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   
venv\Scripts\activate
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run the app**
```bash
streamlit run algo-visualizer.py
```
