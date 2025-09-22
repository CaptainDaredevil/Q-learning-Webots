from controller import Supervisor
import math, random, os, csv, sqlite3
from collections import deque, defaultdict


class LabyrinthRL(Supervisor):

    # ————————————————— Hyperparameters —————————————————
    ACTIONS = [
        ("drive", 10), ("drive", 25),
        ("turn", 30),  ("turn", -30),
        ("turn", 90),  ("turn", -90),
        ("backup", 10)
    ]
    N_ACTIONS = len(ACTIONS)

    STEP_PENALTY     = -0.8          # Step penalty
    SHAPING_BETA     = 10.0          # Reward shaping coefficient
    BACKTRACK_FACTOR = -2.0          # Penalty for moving backward
    GOAL_BONUS       = 300.0         # Bonus for reaching the goal
    R_CLIP           = 500.0         # Maximum reward clipping
    GAMMA            = 0.85          # Discount factor

    ALPHA_MIN = 0.05                 # Lower bound for adaptive α

    EPS_START = 1.0                  # Initial probability of random actions
    EPS_FINAL = 0.05                 # Final probability of random actions
    EPS_DECAY = 0.997                # Decay rate for random actions

    SUCCESS_STREAK_TARGET = 5        # Consecutive goals before freezing training
    EPS_LOCK       = 0.02            # Fixed ε value after freezing
    EPS_LOCK_E     = 0.00            # Fixed ε value after enabling Hyper mode 
    RESCUE_WINDOW  = 30              # Episode window for recovery check
    RESCUE_RATE    = 0.5             # Success rate threshold for recovery
    EPS_RESCUE_INC = 0.10            # Increase of ε during recovery

    MAX_STEPS     = 400              # Maximum steps in one episode
    STUCK_TIME_MS = 800              # Time after which the robot is considered stuck

    DB_FILE, Q_FILE = "training_logs.db", "q_table.csv"  # Database and Q-table files
    SAVE_EVERY      = 250            # Number of episodes between Q-table saves
    GOAL_DIST       = 0.2            # Distance to goal for success

    # ————————————————— Initialization —————————————————
    def __init__(self):
        super().__init__()

        # Get access to the robot and apple
        self.robot  = self.getSelf()
        self.apple  = self.getFromDef("APPLE_Solid")
        self.timestep = int(self.getBasicTimeStep())

        # Store initial robot position
        self.start_trans = list(self.robot.getField("translation").getSFVec3f())
        self.start_rot   = list(self.robot.getField("rotation").getSFRotation())

        # Get motors
        self.left_motor  = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        for m in (self.left_motor, self.right_motor):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        # Get proximity sensors
        self.ps = [self.getDevice(f"ps{i}") for i in range(8)]
        for s in self.ps:
            s.enable(self.timestep)

        # Keyboard initialization
        self.kb = self.getKeyboard()
        self.kb.enable(self.timestep)

        # Q-learning table initialization
        self.q_table, self.visit_counts = self._load_qtable()
        self.training_enabled = True
        self.epsilon = self.EPS_START
        self._changed_states = set()

        # Logging
        self.conn, self.cur = self._init_db()
        self.success_history = deque(maxlen=self.RESCUE_WINDOW)
        self.success_streak  = 0

    # ————————————————— Database and CSV operations —————————————————
    def _init_db(self):
        """Initialize database for saving training info."""
        conn = sqlite3.connect(self.DB_FILE)
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episodes(
                id INTEGER PRIMARY KEY,
                session, steps, reward, min_dist, epsilon, success,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        conn.commit()
        return conn, cur

    def _load_qtable(self):
        """Load Q-table and state visit counts."""
        q  = defaultdict(lambda: [0.0] * self.N_ACTIONS)
        vc = defaultdict(lambda: [0]   * self.N_ACTIONS)
        if not os.path.exists(self.Q_FILE):
            return q, vc
        with open(self.Q_FILE) as f:
            rdr = csv.reader(f)
            next(rdr, None)
            for row in rdr:
                s = tuple(int(x) for x in row[:5])  # State
                q_vals = [float(x) for x in row[5 : 5 + self.N_ACTIONS]]  # Q-values
                
                rest = row[5 + self.N_ACTIONS :]
                try:
                    n_vals = [int(float(x)) for x in rest[: self.N_ACTIONS]]  # Visit counts
                except ValueError:
                    n_vals = [0] * self.N_ACTIONS

                q[s]  = q_vals + [0.0]*(self.N_ACTIONS - len(q_vals))
                vc[s] = n_vals + [0]*(self.N_ACTIONS - len(n_vals))
        return q, vc

    def _flush_qtable(self, force=False):
        """Save Q-table to disk."""
        if not (force or self._changed_states):
            return
        tmp = self.Q_FILE + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            header = ["d","a","f","l","r"] + \
                     [f"q{i}" for i in range(self.N_ACTIONS)] + \
                     [f"n{i}" for i in range(self.N_ACTIONS)]
            w.writerow(header)
            for s, q_vals in self.q_table.items():
                w.writerow(list(s) +
                           [f"{v:.4f}" for v in q_vals] +
                           self.visit_counts[s])
        os.replace(tmp, self.Q_FILE)
        self._changed_states.clear()

    # ————————————————— Geometry and state —————————————————
    @staticmethod
    def _xz_dist(a, b):
        """Calculate Euclidean distance between two points in x-z plane."""
        return math.hypot(a[0]-b[0], a[2]-b[2])

    def _state(self):
        """Determine robot state, including distance to goal and angles."""
        pos  = self.robot.getField("translation").getSFVec3f()
        goal = self.apple.getField("translation").getSFVec3f()

        # Calculate distance to goal and classify it
        d = self._xz_dist(pos, goal)
        dist_bin = 0 if d < 0.2 else 1 if d < 0.5 else 2 if d < 1.0 else 3

        # Calculate relative angle between robot and goal
        dx, dz = goal[0]-pos[0], goal[2]-pos[2]
        yaw = self.robot.getField("rotation").getSFRotation()[3]
        rel = (math.atan2(dz, dx) - yaw + math.pi) % (2*math.pi) - math.pi
        angle_bin = int(((rel + math.pi) / (2*math.pi)) * 8) % 8

        # Read proximity sensors
        front = int(self.ps[7].getValue() > 200 or self.ps[6].getValue() > 200)
        left  = int(self.ps[0].getValue() > 200)
        right = int(self.ps[5].getValue() > 200)

        return (dist_bin, angle_bin, front, left, right)

    # ————————————————— Movement —————————————————
    def _turn_precise(self, deg, vel=6.28):
        """Precise rotation by a given angle."""
        rot_field = self.robot.getField("rotation")
        target = abs(math.radians(deg))
        direction = 1 if deg > 0 else -1

        self.left_motor.setVelocity(-direction * vel)
        self.right_motor.setVelocity(direction * vel)

        acc = 0.0
        prev = rot_field.getSFRotation()[3]
        max_iter = int(target / 0.02) + 20
        for _ in range(max_iter):
            if self.step(self.timestep) == -1:
                break
            cur = rot_field.getSFRotation()[3]
            diff = (cur - prev + math.pi) % (2*math.pi) - math.pi
            acc += abs(diff)
            prev = cur
            if acc >= target:
                break

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.step(self.timestep)

    def _do_action(self, idx):
        """Execute action by index."""
        kind, prm = self.ACTIONS[idx]
        if kind == "drive":
            self.left_motor.setVelocity(6.28)
            self.right_motor.setVelocity(6.28)
            for _ in range(prm):
                if self.step(self.timestep) == -1:
                    break
        elif kind == "turn":
            self._turn_precise(prm)
            self._do_action(0)
        elif kind == "backup":
            self.left_motor.setVelocity(-6.28)
            self.right_motor.setVelocity(-6.28)
            for _ in range(prm):
                if self.step(self.timestep) == -1:
                    break
            self._turn_precise(random.choice([90, -90]))
            self._do_action(0)

    # ————————————————— Q-learning —————————————————
    def choose_action(self, s):
        """Choose action using ε-greedy strategy."""
        if self.training_enabled and random.random() < self.epsilon:
            return random.randrange(self.N_ACTIONS)
        return max(range(self.N_ACTIONS), key=self.q_table[s].__getitem__)

    def update_q(self, s, a, r, s_next):
        """Update Q-table values after performing an action."""
        self.visit_counts[s][a] += 1
        α = max(self.ALPHA_MIN, 1.0 / math.sqrt(self.visit_counts[s][a]))
        best_next = max(self.q_table[s_next])
        self.q_table[s][a] += α * (r + self.GAMMA * best_next - self.q_table[s][a])
        self._changed_states.add(s)

    # ————————————————— Keyboard handling —————————————————
    def _handle_keyboard(self):
        """Handle keyboard events for training control and saving."""
        key = self.kb.getKey()
        if key == -1:
            return
        if key == ord('E'):                       # Hyper mode
            self.training_enabled = False
            self.epsilon = self.EPS_LOCK_E
            print("[Keyboard] → evaluation mode (training frozen)")
        elif key == ord('T'):                     # Resume training
            self.training_enabled = True
            self.epsilon = min(self.epsilon + self.EPS_RESCUE_INC, 0.3)
            print("[Keyboard] → training resumed (ε increased)")
        elif key == ord('S'):                     # Save
            self._flush_qtable(force=True)
            print("[Keyboard] Q-table saved")

    # ————————————————— Episode helper functions —————————————————
    def reset_scene(self):
        """Reset scene to initial position."""
        self.robot.getField("translation").setSFVec3f(self.start_trans)
        self.robot.getField("rotation").setSFRotation(self.start_rot)
        self.robot.resetPhysics()
        for _ in range(5):
            self.step(self.timestep)

    # ————————————————— Main training loop —————————————————
    def run(self, max_sessions=100000):
        """Main training loop with logging and periodic Q-table saving."""
        print(f"[RL] starting {max_sessions} episodes …")
        for ep in range(1, max_sessions + 1):
            self.reset_scene()
            state = self._state()
            prev_dist = self._xz_dist(self.start_trans,
                                      self.apple.getField("translation").getSFVec3f())
            min_dist, steps, ep_reward, stuck_ms = prev_dist, 0, 0.0, 0
            last_pos = self.start_trans
            success = False

            while self.step(self.timestep) != -1 and steps < self.MAX_STEPS:
                self._handle_keyboard()

                a = self.choose_action(state)
                self._do_action(a)
                steps += 1

                pos = self.robot.getField("translation").getSFVec3f()
                dist = self._xz_dist(pos, self.apple.getField("translation").getSFVec3f())
                min_dist = min(min_dist, dist)
                next_state = self._state()

                # Reward shaping
                delta = prev_dist - dist
                shaping = self.SHAPING_BETA * delta if delta > 0 else \
                          self.BACKTRACK_FACTOR * abs(delta)
                reward = self.STEP_PENALTY + shaping
                reward = max(-self.R_CLIP, min(self.R_CLIP, reward))

                # Stuck detection
                moved = self._xz_dist(pos, last_pos) > 0.01
                if not moved:
                    stuck_ms += self.timestep
                    if stuck_ms >= self.STUCK_TIME_MS:
                        self._do_action(6)
                        reward -= 2.0
                        stuck_ms = 0
                else:
                    stuck_ms = 0
                last_pos = pos

                if self.training_enabled:
                    self.update_q(state, a, reward, next_state)

                state, prev_dist = next_state, dist
                ep_reward += reward

                # Goal check
                if dist < self.GOAL_DIST:
                    success = True
                    if self.training_enabled:
                        self.update_q(state, a, self.GOAL_BONUS, next_state)
                    ep_reward += self.GOAL_BONUS
                    break

            # Logging
            if ep >= 100:
                self.cur.execute(
                    "INSERT INTO episodes(session,steps,reward,min_dist,epsilon,success) "
                    "VALUES(?,?,?,?,?,?)",
                    (ep, steps, ep_reward, min_dist, self.epsilon, int(success))
                )
                self.conn.commit()

            # Epsilon update
            if self.training_enabled and self.epsilon > self.EPS_FINAL:
                self.epsilon = max(self.EPS_FINAL, self.epsilon * self.EPS_DECAY)

            # Recovery logic
            self.success_history.append(int(success))
            self.success_streak = self.success_streak + 1 if success else 0

            if self.success_streak >= self.SUCCESS_STREAK_TARGET and self.training_enabled:
                self.training_enabled = False
                self.epsilon = self.EPS_LOCK
                print(f"▶ Policy frozen after {self.SUCCESS_STREAK_TARGET} consecutive goals")

            if not self.training_enabled and len(self.success_history) == self.RESCUE_WINDOW:
                rate = sum(self.success_history) / self.RESCUE_WINDOW
                if rate < self.RESCUE_RATE:
                    self.training_enabled = True
                    self.epsilon = min(self.epsilon + self.EPS_RESCUE_INC, 0.3)
                    print("✘ Performance drop — training resumed")
                self.success_history.clear()

            # Periodic Q-table save
            if ep % self.SAVE_EVERY == 0:
                self._flush_qtable()
                print(f"        Q-table saved ({len(self.q_table)} states)")

            status = "GOAL" if success else "timeout"
            print(
                f"Ep {ep:4d}: {status:7s} | steps={steps:3d} | "
                f"R={ep_reward:8.1f} | minD={min_dist:.2f} | ε={self.epsilon:.3f}"
                f"{'' if self.training_enabled else ' [eval]'}"
            )

        self._flush_qtable(force=True)
        print("[RL] training finished — table saved.")


if __name__ == "__main__":
    LabyrinthRL().run()
