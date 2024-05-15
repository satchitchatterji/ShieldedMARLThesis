% actions
action(0)::action(left);
action(1)::action(right);
action(2)::action(up);
action(3)::action(down);
action(4)::action(stay).

sensor_value(0)::sensor(left).
sensor_value(1)::sensor(right).
sensor_value(2)::sensor(up).
sensor_value(3)::sensor(down).
sensor_value(4)::sensor(stag_near_self).
sensor_value(5)::sensor(stag_near_other).

% hunt stag if it is near and there is another agent
hunt :- action(Dir), sensor(Dir).

% move_away :- action(Dir1), sensor(Dir2), Dir1 \= Dir2.

% hunt if stag is near and there is another agent
safe_next :- \+hunt.
