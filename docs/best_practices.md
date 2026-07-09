# Best Practices

This section collects best practices to help you write clean, efficient, and
reusable code when working with REAF.

--------------------------------------------------------------------------------

## Make Specs Available Immediately After Initialization

Your environment's **`action_spec()`** and **`timestep_spec()`** should be fully
defined and accessible immediately after the environment object is created,
without needing to call `reset()` or other methods first.

**Why is this important?** Many downstream tools, such as run loops, loggers,
and replay buffers, inspect these specifications during their own initialization
to allocate resources and validate configurations. Making specs available on
construction also simplifies debugging and unit testing.

**Example:**

```py

# Create the environment object.
# At this point, no hardware resources are acquired, but the specs are ready.

my_env = env_builder.build()

# The specs can be immediately queried and used.
print("Timestep Spec:", my_env.timestep_spec())
print("Action Spec:", my_env.action_spec())
```

## Manage Resource Lifecycles Explicitly

Connecting to hardware is often a slow and resource-intensive operation. For
this reason, you should **never** acquire resources (like opening a network
connection or initializing hardware drivers) inside your environment's
`__init__` method.

Instead, separate the **construction** of your environment from the
**connection** to its resources. We recommend implementing explicit `connect()`
and `close()` (or `initialize()` and `finalize()`) methods in your
`DeviceCoordinator`. This approach gives you full control over when and how
resources are managed, which is crucial for both testing and deployment.

**Why is this important?** Explicitly managing resources makes your environment
more flexible and easier to test. You can instantiate it in a unit test without
needing real hardware, or manage connections differently depending on the
deployment context (e.g., in a simulation vs. on a physical robot).

**Example:**

```py

# 1. Create the coordinator and environment objects.
#    No resources have been acquired yet.
device_coordinator = MyDeviceCoordinator()
my_env = env_builder.build(device_coordinator=device_coordinator)

try:
    # 2. Explicitly connect to the robot hardware.
    print("Connecting to robot...")
    device_coordinator.connect()
    print("Connection successful.")

    # 3. Now, work with the environment.
    timestep = my_env.reset()
    while not timestep.last():
        action = my_policy.act(timestep)
        timestep = my_env.step(action)

finally:
    # 4. Ensure resources are always released.
    print("Closing connection...")
    device_coordinator.close()
```
