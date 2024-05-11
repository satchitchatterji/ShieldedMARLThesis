
% actions
action(0)::action(stag);
action(1)::action(hare).

sensor_value(0)::sensor(stag).
sensor_value(1)::sensor(hare).

unsafe_next :- action(hare).
safe_next:- \+unsafe_next.
