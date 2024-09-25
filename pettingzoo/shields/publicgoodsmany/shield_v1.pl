% actions
action(0)::action(defect);
action(1)::action(cooperate).

sensor_value(0)::sensor(cooperated).

unsafe_next :- action(defect), sensor(cooperated).
unsafe_next :- action(cooperate), \+sensor(cooperated).
safe_next:- \+unsafe_next.