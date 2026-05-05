# Action_Replicant
End-to-End pipeline to collect, label, and train an agent to emulate player actions in an FPS/TPS

Completed steps:
- Data collection
- BoundingBox Creation
- BoundingBox Labeling
- Teacher Training - RFDETR-Patches
- Teacher Inference
    - Batch inference for student dataset
    - "Real-time" inference for performance evaluation

WIP Steps:
- Move class names into a seed file in code/data
    - Standardizes class references and reduces the need for it to be coded in everywhere
    - Also makes it easier for others to change the code for their custom replicant pipelines
- Image parsed features for Action-prediction agent
- Improvements to Repo
    - Pipeline Diagram(s) - Mermaid
    - Visualization of RFDETR & YOLO performance
    - RFDETR & YOLO aggregate performance (Avg Precision: IoU All/Small [0.5:0.95, 0.5, 0.75])

Future Steps:
- Create Action-Replicant Library
- Create Action-Replicant training loop
- Evaluate Action-Replicant


Usage of this pipeline to create agents that play Helldivers II violates TOS.
The goal of this project is to:
- Showcase skills
- Practice Computer Vision, Deep-learning & Reinforcement Learning concepts
- Develop an agent that can predict my actions with recorded inputs
    - The agent will not be deployed in a capacity that would violate Helldivers II TOS
