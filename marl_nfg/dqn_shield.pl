% actions
action(0)::action(cooperate);
action(1)::action(defect).


% states (discretized)
sensor_value(0)::defection(agent1).
sensor_value(1)::defection(agent2).
sensor_value(2)::agentname(agent1).
sensor_value(3)::agentname(agent2).

% transition
% it is unsafe to cooperate if the other agent defected in the previous round
% unsafe_next :- action(cooperate), defection(X), agentname(X).
% it is unsafe to not cooperate if the other agent cooperated in the previous round
% unsafe_next :- action(defect), \+defection(X), agentname(X).
% it is unsafe to cooperate twice in a row
% unsafe_next :- action(cooperate), \+action(defect).

unsafe_next:- action(cooperate).

safe_next:- \+unsafe_next.
% safe_action(A):- action(A).