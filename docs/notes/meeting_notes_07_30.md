### Meeting Notes
- Visualization dashboard
	- graphs routes, 
- Distance short (swap in same town, postcode) --> no alert
- Same pharmacy --> no alert
- Hospital - Hospital --> alert

- operator (human/agent) in loop as input to spot patterns --> add to pattern database

- 3000 per week in alerts in Bulgaria, most are errors, 70% from 1 software company
- Certain Bad Actors (Blacklist) -> Trust score debuff, a priori high trust score
- Germany 35000 per week
- 160 million scans per year
- < 0.5% --> aim 0.05% of alerts
- layered system -> system alerts -> database rules filter alerts --> trace pack history

Benchmark 95% sold within 3 months from OBP to Wholesalers
Dependent on Organization, time, medicine type, medicine form (tablets, syringes)
- Human in the loop 
- How to define rules
	- data fields
	- User in the loop (which field)
- AlertLabeler

### AlertLabeler Plan
Introduction: Borrowed from typical Human-in-the-loop systems for financial fraud detection, the idea is as follows: We start with a small set of simple rules (fixed parameters, defined in .json file, with fields such as location, distance, time, type of organization, type of medicine). For instance, according to cooperator BDA official, if 2 pharmacies in very close physical proximity (same town, same zip code) engage in a transaction, it should be a false alert (safe). Whereas, if a wholesaler tag scans a highly valuable medicine at 11:00pm, this is highly suspicious and should be alert (unsafe). The "human" user receives a queue of alerts, for each alert the human decides whether it is valid or not . (If convergence speed is too slow, we can add parameters to give gradients to the human's judgment, ex: distance too small to be alert, same location etc) to speed up the learning process. The parameters are updated or rules are adapted based on the running frequency of human feedback of "safe" or "unsafe". 
Create a simple cli-program that integrates with the simulator and does the following:
Frontend: At each tick, present a queue of alerts to the "human in the loop". The "human in the loop (hitl)" is actually a program containing a series of preprogrammed rules. The goal is for the system to learn these rules via feedback only, to demonstrate that the mechanism is capable of converging its behavior to that of the HITL. ...