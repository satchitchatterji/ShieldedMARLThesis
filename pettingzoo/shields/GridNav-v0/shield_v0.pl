% actions
action(0)::action(down);
action(1)::action(left);
action(2)::action(up);
action(3)::action(right).

sensor_value(0)::sensor(down).
sensor_value(1)::sensor(left).
sensor_value(2)::sensor(up).
sensor_value(3)::sensor(right).

all_sensors :- sensor(down), sensor(left), sensor(up), sensor(right).

unsafe_next :- action(X), sensor(X), \+all_sensors.
safe_next:- \+unsafe_next.