# NFL Game Simulator

Each of the boxes below has an ML model associated with it that takes in features of the game state, offensive and defensive teams and players. 
Here's the flow for the simulation:

```mermaid
flowchart 
aa[Start Game] --> A
A[Update Game State] --> B(Choose play type)
B --> C{Run Play}
B --> D{Pass Play}
B --> E{Timeout}
B --> F{QB Spike}
B --> G{QB Kneel}
C --> H[Choose Rusher]
H --> I{Yards Rushed}
D --> J[Choose Passer]
J --> K[Choose Receiver]
K --> L[Sim Air Yards]
L --> M[Sim Completion]
M --> N[Sim Yards After Catch]
N --> A
E --> A
F --> A
G --> A
I --> A
```