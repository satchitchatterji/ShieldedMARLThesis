% actions
action(0)::action(defect);
action(1)::action(cooperate).

sensor_value(0)::sensor(cooperated).

unsafe_next :- action(defect).
safe_next:- \+unsafe_next.