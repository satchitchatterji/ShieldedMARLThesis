% shield to filter wandb runs for new cartsafe sensor
% actions
action(0)::action(left);
action(1)::action(right).

sensor_value(0)::sensor(cost).
sensor_value(1)::sensor(xpos).
sensor_value(2)::sensor(left).
sensor_value(3)::sensor(right).

unsafe_next :- action(X), sensor(X), sensor(cost), sensor(xpos).
safe_next:- \+unsafe_next.