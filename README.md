# COBALT: Crowdsourcing Robot Learning via Cloud-Based Teleoperation with Smartphones

Ayush Agarwal*, Ansh Gandhi*, Jeremy A. Collins, Omar Rayyan, Aryan Sarswat, Ranjani Koushik, Masoud Moghani, Ajay Mandlekar, Animesh Garg

[![Website](https://img.shields.io/badge/Website-cobalt--teleop.github.io-0a84ff?logo=google-chrome&logoColor=white&style=flat)](https://cobalt-teleop.github.io/)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2506.14198-b31b1b.svg?logo=arXiv&logoColor=white&style=flat)](https://arxiv.org/abs/2506.14198) -->
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white&style=flat)](https://www.python.org)
![license](https://img.shields.io/github/license/pairlab/AMPLIFY?style=flat&cacheSeconds=1)

This repo contains the official implementation for COBALT: Crowdsourcing Robot Learning via Cloud-Based Teleoperation with Smartphones. COBALT is a platform for collecting robot demonstrations through smartphone-based teleoperation. It enables crowdsourced data collection for robot learning by allowing users to control robots remotely using their mobile devices. It is easily deployable to the cloud.

## Related Links

* [Project website](https://cobalt-teleop.github.io/)
* [COBALT Isaac Lab repo](https://github.com/pairlab/cobalt-isaaclab/)
* [COBALT Mobile App repo](https://github.com/pairlab/cobalt-mobile-app/)

## Quick Start

### Prerequisites

- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)
- Google Chrome

### Installation

COBALT consists of separate backend and frontend services that need to be configured independently.

#### 1. Clone submodules

```bash
git submodule update --init --recursive
```

#### 2. Backend Setup

**Default (no GPU, Robosuite):**
```bash
cd backend
./setup.sh
python setup_backend.py
```

**Optional IsaacLab setup (GPU required):**
```bash
cd backend
./setup.sh --with-isaaclab
python setup_backend.py
```

#### 3. Frontend Setup

```bash
cd ../frontend
python setup_frontend.py
```

> **Note:** Setup scripts automatically detect your public IP address for nginx configuration. To use a custom domain, pass it as an argument: `python setup_backend.py --domain your-domain.com` or `python setup_frontend.py --domain your-domain.com`

### Firewall Configuration

COBALT uses nginx as a reverse proxy to route requests to different services. Configure your firewall based on your deployment scenario:

```bash
# Allow traffic to frontend (optional if using localhost)
sudo ufw allow 80

# Allow traffic to backend
sudo ufw allow 8080

# Allow traffic to be forwarded from nginx to media-server
sudo ufw allow from 172.18.0.0/16 to any port 5001 proto tcp

# Allow UDP connections on these ports for WebRTC communication
sudo ufw allow 49152:65535/udp
```

> **Port Customization:** External ports can be modified in `backend/nginx/backend.conf.template` and `frontend/nginx/frontend.conf.template`. Internal service ports are configured in `backend/docker.env` and `frontend/docker.env`.

## Usage

### Starting the Services

Start both services in separate terminal windows:

**Backend Service:**
```bash
cd backend
sudo TASK=lift SIMULATOR=robosuite docker compose --env-file docker.env up
```

**Backend Service (IsaacLab / GPU):**
```bash
cd backend
sudo TASK=lift SIMULATOR=isaaclab docker compose --env-file docker.env up
```

**Frontend Service:**
```bash
cd frontend
sudo docker compose up
```

#### Supported Simulators and Tasks

- **Simulators:** `robosuite`, `isaaclab`
- **Task Lists:** See `isaaclab_tasks.json` and `robosuite_tasks.json` for available tasks. For bimanual tasks, you will need two devices to connect to the same session before starting teleoperation.
- **Upgrade Path:** You can start with the default Robosuite setup and run `./setup.sh --with-isaaclab` later to add IsaacLab support.

### Teleoperation Method

1. **Download Mobile App**
   - iOS: [Apple Test Flight](https://testflight.apple.com/join/p51bs9Wu)
   - Android: [Android APK](https://drive.google.com/file/d/1SGaMf3Tlsd9CPKaIdOBfprL2U3itID2a/view?usp=drive_link)
2. **Configure Mobile App:**
   - Open app settings (gear icon on top-right)
   - Set **Server IP Address** to your host machine's IP
   - Set **Server Port** to `8080`
   - Select any task (host machine determines actual task)
3. **Connect Mobile Device:**
   - Return to main screen
   - Tap "Connect"
4. **Start Video Stream:**
   - Navigate to `http://localhost:3000` in your browser
   - Enter your host machine's IP address
   - Enter the Session ID displayed on your mobile app
   - Click "Start Stream"

### Teleoperating with Keyboard
To teleoperate with a keyboard, first start the backend service as described above. Then run:
```bash
python backend/teleop-server/RobotTeleop/clients/client.py
```
Notes:
- Keyboard teleop client depends on `pynput` and `websockets` (included in `backend/teleop-server/requirements-core.txt`).
- `RobotTeleop/clients/keyboard.py` requires running the keyboard client as root. If needed, use `sudo`.
- Useful client flags:
  - `--host` (default: `localhost`)
  - `--port` (default: `8080`)
  - `--task` (default: `test`; should match your backend task route)
  - `--session_id` (attach to an existing session)
  - `--sim-type` (default: `bimanual`)

Example (local machine):
```bash
python backend/teleop-server/RobotTeleop/clients/client.py --host localhost --port 8080 --task lift
```

COBALT also supports mixing devices (for example, smartphone + keyboard in bimanual sessions).

### Adding New Robots and Simulators
To add support for a new robot or simulator, update the extension points below.

#### 1) Add a new robot class
- Add a robot implementation under `backend/teleop-server/RobotTeleop/robots/` (typically inheriting Teleop/IK/OSC robot interfaces).
- Import it in `backend/teleop-server/RobotTeleop/robots/__init__.py` so it registers via the robot metaclass/registry in `robots/interfaces/teleop_robot.py`.
- Reference the robot class name in the appropriate server config (for example in `base_config_robosuite.py`, `base_config_isaac.py`, or a new config file).

#### 2) Add a new simulator class
- Implement a simulator under `backend/teleop-server/RobotTeleop/simulators/` by inheriting `BaseSimulator` (`simulators/base_simulator.py`).
- Register it in `simulators/simulator_factory.py` via `SimulatorFactory.register_simulator("<name>", YourSimulatorClass)`.
- Make sure `config.simulator.name` matches the registered name.

#### 3) Add/update config and task mappings
- Add task definitions in `backend/teleop-server/RobotTeleop/configs/tasks/` (JSON).
- Update or add a server config in `backend/teleop-server/RobotTeleop/configs/` to map:
  - simulator name
  - robot type(s)
  - task metadata (`task`, `num_device`, timeout)
- Set environment variables at startup, for example:
  - `SIMULATOR=robosuite` or `SIMULATOR=isaaclab`
  - `TASK=<task_name>`

### Adding New Tasks
To add new tasks, you will need to add new environments to the underlying packages (e.g. `src/robosuite`, `src/isaaclab`, etc.). After doing this, ensure to add the new task to the appropriate tasks JSON file (e.g. `backend/teleop-server/RobotTeleop/configs/tasks/robosuite_tasks.json`, `backend/teleop-server/RobotTeleop/configs/tasks/isaaclab_tasks.json`) so that COBALT recognizes the new task.

## Post-Processing Demonstrations

**Merge multiple demonstration files:**
```bash
cd backend/teleop-server/RobotTeleop/scripts
python post_process_data.py --input-folder <INPUT_FOLDER> --output-file <OUTPUT_FILE>
```

**Generate demonstration metrics:**
```bash
cd backend/teleop-server/RobotTeleop/scripts
python demo_metrics.py --input-file <INPUT_FILE> --output-file <OUTPUT_FILE>
```

## Advanced Configuration

### TURN Server Setup (Optional)

For better connectivity across NAT/firewall boundaries, you can set up a TURN server:

#### 1. Deploy CoTURN Server

```bash
cd backend
python coturn/setup_coturn.py
sudo docker run -d --network=host -v $(pwd)/coturn/custom.conf:/etc/coturn/turnserver.conf coturn/coturn
```

#### 2. Configure Frontend TURN Credentials

Create `frontend/.env`:
```env
REACT_APP_TURN_URL=<TURN_URL>
REACT_APP_TURN_USERNAME=<USERNAME>
REACT_APP_TURN_PASSWORD=<PASSWORD>
```

> **Note:** Also add these credentials to the `iceServers` configuration in `VideoChat.tsx`

#### 3. Configure Media Server TURN Credentials

Create `backend/media-server/.env`:
```env
TURN_SERVER_1_URL=<TURN_URL>
TURN_SERVER_1_USERNAME=<USERNAME>
TURN_SERVER_1_PASSWORD=<PASSWORD>
```

## Troubleshooting

### Common Issues

**Connection Problems:**
- **Cause:** Blocked ports on host machine
- **Solution:** Ensure firewall ports are open (see [Firewall Configuration](#firewall-configuration))
- **Note:** Docker does NOT automatically open ports - manual firewall configuration is required

**Mobile App Won't Connect:**
- Verify host machine IP address is correct
- Check that backend service is running on port 8080 and is **accessible from another device**

**Video Stream Not Loading:**
- Confirm frontend service is running
- Check Session ID matches between mobile app and web interface
- Verify media server is accessible (check Docker logs)

**Docker Issues:**

*NVIDIA GPU Driver Error:*
```
Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]
```

- **Cause:** NVIDIA Container Toolkit is not installed
- **When it appears:** IsaacLab mode (`--profile isaaclab`)
- **Solution:** Install and configure the toolkit:

```bash
# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use it
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

**Last Updated:** May 15, 2026
