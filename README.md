# AI Engineering Project Disruptive GmbH

A cutting-edge AI Engineering Project Disruptive GmbH that combines the power of Neo4j Knowledge Graphs and Retrieval-Augmented Generation (RAG) to deliver highly contextual and intelligent responses. This project idea leverages the strengths of graph-based knowledge representation and retrieval-based AI to enhance chatbot performance and ensure accurate, relevant, and dynamic interactions.

### Please follow the Postman collections

<h3>Features</h3>
<ol>
  <li><b>Neo4j Knowledge Graph Integration:</b> Organizes and manages structured knowledge for enhanced context-awareness.</li>
  <li><b>Retrieval-Augmented Generation (RAG):</b> Employs a hybrid approach by retrieving knowledge graph data and combining it with generative AI for robust responses.</li>
  <li><b>Scalable Architecture:</b> Designed to scale with data size and complexity for enterprise applications.</li>
  <li><b>Dynamic Querying:</b> Uses Cypher queries to extract relevant knowledge graph data in real-time.</li>
  <li><b>API Integration:</b> Includes APIs for seamless integration with web and mobile applications.</li>
</ol>


<h3>Prerequisites</h3>
<ul>
  <li>Python 3.11</li>
  <li>Libraries Installation</li>
  <li>Neo4j</li>
</ul>


<h3>Installation</h3>
<ul>
  <li>Clone the repository:</li>

  ```bash
git https://github.com/Bikas0/AI-Engineering-Project-Disruptive-GmbH.git
cd  AI-Engineering-Project-Disruptive-GmbH
```
</ul>

<h3>Docker Compose</h3>

```bash
docker-compose up --build -d
```

<h3>Usage</h3>

<b>Interact with the API:</b>

```bash
http://0.0.0.0:5507
```

<h3>Understanding APIs</h3>
<ol>
  <li><b>Clear Database:</b> User can clear the database using the code.</li>

  ```bash
  http://0.0.0.0:5507/clear-database
  ```

  <li><b>Upload Data in the Neo4j database:</b> It takes a lot of time to convert the raw text into a Neo4j database while creating the entities.</li>

  ```bash
  http://0.0.0.0:5507/embedding-data
  ```

  <li><b>Chat with Model:</b> Delivers the final response to the user via API.</li>

  ```bash
  http://0.0.0.0:5507/chat
  ```

</ol>

<h3>Acknowledgments</h3>
<ul>
  <li>Neo4j</li>
  <li>LangChain</li>
  <li>Llama3.2</li>
</ul>