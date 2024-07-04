
% actions
action(0)::action(stag);
action(1)::action(hare).

sensor_value(0)::sensor(stag_diff).
sensor_value(1)::sensor(hare_diff).

unsafe_next :- action(stag), sensor(stag_diff).
unsafe_next :- action(hare), sensor(hare_diff).

safe_next:- \+unsafe_next.
