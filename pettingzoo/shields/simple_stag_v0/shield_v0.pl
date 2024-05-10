
% actions
action(0)::action(stag);
action(1)::action(hare).

% 30 sensors for obstacles
sensor_value(0)::sensor(stag).
sensor_value(1)::sensor(hare).

unsafe_next :- action(hare).
% safe_next :- action(stag).
safe_next:- \+unsafe_next.
