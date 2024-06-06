
% actions
action(0)::action(defect);
action(1)::action(cooperate).

sensor_value(0)::sensor(defect).
sensor_value(1)::sensor(cooperate).
sensor_value(2)::sensor(none).

unsafe_next :- action(defect).
safe_next:- \+unsafe_next.
