## Modules and Event Sequence:

```mermaid
sequenceDiagram
    participant U as User
    box Agent
        participant A as Agent
        participant S as Short Term Memory
        participant P as User Profile
        participant H as Chatting History
        participant L as Long Term Memory
    end
    box Tools
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

This design is motivated by the book "Thinking, Fast and Slow", which discusses two systems:
  + System 1 is fast and intuitive, operates automatically and unconsciously, and helps us make quick judgments and react to our environment. 
  + System 2 is slow, conscious, and used for complex tasks like problem-solving or decision-making. It requires effort, enables logical reasoning, and systematic analysis. ==> Tools
* Agent is 

- System1 is Agent which handles
  + Intuitive/easy question 
  + Coordination
  + Communication
  + ...
- System2 consists of Tools which handles complex tasks:
  + Math
  + Information retrieval
  + ...







