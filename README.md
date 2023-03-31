## Modules and Event Sequence:

```mermaid
sequenceDiagram
    participant U as User
    box System1
        participant A as Agent
        participant S as Short Term Memory
        participant P as User Profile
        participant H as Chatting History
        participant L as Long Term Memory
    end
    box System2
        participant M as ToolManager
        participant ML as MachineLearningTool
        participant VA as VacationTool
    end
    U-->>A: query=What is the loss for my ML model ABC?
    par Agent self asking
        A-->>S: Request context
        S-->>P: Request user profile
        P-->>S: Return user profile
        S-->>H: Request chatting history
        H-->>S: Return chatting history
        S-->>L: Request related sentences
        L-->>S: Return related sentences
        S-->>A: Return context
        A-->>A: Can I answer the quest given the context?. No
    end
    A-->>M: Request right tools (query)
    par Tool asking
        M-->>ML: Can you handle? (query)
        ML-->>M: Yes, (answer: the loss is 0.96)
        M-->>VA: Can you handle? (query)
        VA-->>M: No
    end
    M-->>A: Return MachineLearningTool's answer
    par Agent reflection
        A-->>A: Can can answer the query given the context and tools' answer?. Yes
        A-->>U: The loss for ML model ABC is 0.96
    end
```

The book "Thinking, Fast and Slow" discuss two systems:
  + System 1 is fast and intuitive, operates automatically and unconsciously, and helps us make quick judgments and react to our environment.
  + System 2 is slow, conscious, and used for complex tasks like problem-solving or decision-making. It requires effort, enables logical reasoning, and systematic analysis.

Details in each components
* Agent:
* Short Term Memory:
* User Profile:
* Chatting History:
* Long Term Memory:
* ToolManager:
* Machinelearning Tool:


### [Demo recording](https://www.dropbox.com/s/mdwcg1f8uali5ma/Screen%20Recording%202023-03-30%20at%2011.36.05%20AM.mov?dl=0)
