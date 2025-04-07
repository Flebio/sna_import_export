# 🌐 International Trade Network Analysis  
### Understanding Global Economic Interactions

This project explores the structure and dynamics of international trade using network analysis. By modeling countries and their trade relationships as a directed graph, we identify key economic players, visualize global trade flows, and uncover potential vulnerabilities in the system.

---

## 🧠 Project Motivation

International trade is a cornerstone of the global economy. This project aims to:

- Understand the **structure** of global trade networks
- Identify **central countries** and critical connections
- Assess potential **vulnerabilities** in trade routes
- Provide insights useful for **policy makers** and **economists**

We focus on both **imports** and **exports** as separate directed networks to offer a clearer view of their structures and influence.

---

## 📦 Dataset

- **Source**: [CIA World Factbook](https://www.cia.gov/the-world-factbook/)
- **Format**: JSON file (`countries.json`)
- **Nodes**: Countries
- **Edges**: Directed, representing import/export relationships
- **Weights**: Percentage of total trade with a partner country

### Data Processing

- Parsed and cleaned using Python (`json`, `pandas`)
- Mapped trade partners and percentages into edge lists
- Removed incomplete or redundant nodes (e.g., Guadeloupe, Martinique)
- Constructed two separate directed graphs:
  - **Exports Network**: 207 nodes, 972 edges
  - **Imports Network**: 207 nodes, 1003 edges

---

## 🧪 Methodology

### Tools Used
- **Python**
  - `pandas`, `json`, `networkx`, `plotly`
- **Plotly**: For interactive visualizations
- **NetworkX**: Graph creation and analysis

---

## 👥 Authors

- **Fabio Zanotti** – Artificial Intelligence  
- **Edoardo Conca** – Artificial Intelligence  
- **Antonio Morelli** – Artificial Intelligence  



