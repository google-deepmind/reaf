# Overview

## What is REAF?

REAF, which stands for Robotics Environment Authoring Framework, is a tool
designed to streamline the creation of robotics environments. It ensures that
all created environments are compliant with the
[GDM Robotics Environment interface](https://github.com/google-deepmind/gdm_robotics/blob/main/src/gdm_robotics/interfaces/environment.py).

--------------------------------------------------------------------------------

## Why REAF?

The concept of an "Environment" is a fundamental abstraction in Reinforcement
Learning, representing the world with which an "agent" interacts. While this
abstraction is widely used, most implementations are ad-hoc, leading to several
challenges that limit scalability and collaboration.

REAF was developed to address these challenges by providing a standardized
structure for authoring robotics environments. It is designed to be scalable and
flexible, supporting a wide variety of robots and tasks. By offering a common
framework, REAF aims to reduce code complexity, improve debuggability, and make
it easier for new users to get started.

--------------------------------------------------------------------------------

## Core Concepts

REAF's design is guided by a few core principles that you will see reflected
throughout the framework and its documentation:

-   **Separation of Embodiment and Task**: REAF enforces a clear distinction
    between the embodiment (the physical robot or simulation) and the task (the
    goal the embodiment is trying to achieve). This separation allows for
    greater modularity and reusability. Embodiments, which are typically stable,
    can be developed and maintained independently from tasks, which are more
    dynamic and experimental.

-   **Adherence to a Standard Interface**: All environments created with REAF
    adhere to the
    [GDM Environment interface](https://github.com/google-deepmind/gdm_robotics/blob/main/src/gdm_robotics/interfaces/environment.py).
    This ensures consistency and interoperability across different projects and
    teams.

-   **Integrated Workflow**: REAF provides built-in support for essential
    functionalities like **resets** and **logging**, which are crucial for
    conducting robotics research in a systematic and reproducible manner.

--------------------------------------------------------------------------------

## Getting started

You can install `reaf` from PyPI using `pip`:

```bash
pip install reaf
```

To learn more about the architecture and design of REAF, we recommend reading
the [Core concepts](core_concepts.md) document. For hands-on experience, you can
follow our [tutorials](tutorials/index.md).

--------------------------------------------------------------------------------

## Licence and Disclaimer

Copyright 2026 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this file except in compliance with the Apache 2.0 license. You may
obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
