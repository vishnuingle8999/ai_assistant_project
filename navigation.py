import json
import math
from urllib.request import Request, urlopen
from urllib.error import URLError


class NavigationAssistant:
    def __init__(self):
        # Preset locations (Replace with your real coordinates from Google Maps)
        self.presets = {
            "home": {"lat": 19.0760, "lon": 72.8777, "name": "Home"},
            "office": {"lat": 19.0650, "lon": 72.8680, "name": "Office"},
            "college": {"lat": 19.0450, "lon": 72.8890, "name": "College"},
            "hospital": {"lat": 19.0820, "lon": 72.8810, "name": "Hospital"},
        }

        # Core safety and accuracy parameters
        self.max_accuracy = 25.0     # Ignore noisy GPS readings worse than 25 meters
        self.turn_radius = 12.0      # Announce the next turn when 12 meters away
        self.off_route_limit = 35.0  # Auto-reroute if user drifts 35m the wrong way
        self.timeout = 4.0           # Hard network cutoff in seconds

        # Active route memory
        self.steps = []
        self.step_index = 0
        self.active_dest = None
        self.min_dist = float("inf")
        self.halfway_spoken = False
        self.last_pos = None

    @staticmethod
    def get_distance(lat1, lon1, lat2, lon2):
        # Fast flat-earth distance formula in meters
        rad = math.pi / 180.0
        x = (lon2 - lon1) * rad * math.cos((lat1 + lat2) * 0.5 * rad)
        y = (lat2 - lat1) * rad
        return math.hypot(x, y) * 6371000.0

    @staticmethod
    def _clean_instruction(m_type, modifier, street, dist, dest_name):
        # Convert compass words into simple body-relative directions
        if m_type == "arrive":
            return f"You have arrived at {dest_name}."

        road = f" onto {street}" if street else ""
        if "left" in modifier or "right" in modifier:
            action = f"Turn {modifier}"
        elif modifier == "uturn":
            action = "Make a U-turn"
        else:
            action = "Walk straight"

        return f"{action}{road} for {dist} meters."

    def process_command(self, lat, lon, command_text):
        # Clean the spoken text
        cmd = command_text.lower().strip()

        # 1. Stop / Cancel Command
        if any(w in cmd for w in ("stop", "cancel", "clear", "exit")):
            self.steps = []
            self.active_dest = None
            return {"status": "cancelled", "speech": "Navigation stopped."}

        # 2. Repeat / Status Command
        if any(w in cmd for w in ("repeat", "where", "status", "distance", "far")):
            if not self.steps or self.step_index >= len(self.steps):
                return {"status": "idle", "speech": "No active route. Say take me home or office."}
            step = self.steps[self.step_index]
            dist = int(self.get_distance(lat, lon, step["lat"], step["lon"]))
            return {
                "status": "active",
                "speech": f"Heading to {self.active_dest}. {step['instruction']} {dist} meters to next turn."
            }

        # 3. Help Command
        if "help" in cmd or "command" in cmd:
            return {
                "status": "help",
                "speech": "Say take me home, office, college, hospital, repeat, or stop navigation."
            }

        # 4. Start Navigation to Preset Destination
        return self.start_route(lat, lon, cmd)

    def start_route(self, start_lat, start_lon, cmd):
        target = next((v for k, v in self.presets.items() if k in cmd), None)
        if not target:
            return {
                "status": "error",
                "speech": f"I heard {cmd}. Please say Home, Office, College, Hospital, or Repeat."
            }

        url = (
            f"http://router.project-osrm.org/route/v1/foot/"
            f"{start_lon},{start_lat};{target['lon']},{target['lat']}"
            f"?overview=false&steps=true"
        )

        try:
            req = Request(url, headers={"User-Agent": "VisionNav/1.0"})
            with urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            if data.get("code") != "Ok":
                return {"status": "error", "speech": "Could not calculate a walking path."}

            route = data["routes"][0]
            clean_steps = []

            for s in route["legs"][0]["steps"]:
                dist = int(s.get("distance", 0))
                m_type = s["maneuver"].get("type", "walk")
                if dist < 3 and m_type != "arrive":
                    continue

                modifier = s["maneuver"].get("modifier", "straight")
                street = s.get("name", "").strip()
                lon, lat = s["maneuver"]["location"]

                text = self._clean_instruction(m_type, modifier, street, dist, target["name"])
                clean_steps.append({"instruction": text, "lat": lat, "lon": lon, "distance": dist})

            self.steps = clean_steps
            self.step_index = 0
            self.active_dest = target["name"]
            self.min_dist = float("inf")
            self.halfway_spoken = False

            first_msg = self.steps[0]["instruction"] if self.steps else "Route ready."
            total_m = int(route["distance"])
            return {
                "status": "started",
                "destination": target["name"],
                "distance_to_turn": self.steps[0]["distance"] if self.steps else 0,
                "speech": f"Routing to {target['name']}, {total_m} meters away. {first_msg}"
            }

        except (URLError, KeyError, IndexError, TimeoutError):
            return {"status": "error", "speech": "Routing service is unreachable right now."}

    def update_position(self, lat, lon, accuracy=10.0):
        # Do nothing if no route is active
        if not self.steps or self.step_index >= len(self.steps):
            return {"status": "idle", "speech": None}

        # 1. Ignore inaccurate GPS jumps (> 25m)
        if accuracy > self.max_accuracy:
            return {"status": "low_gps_accuracy", "speech": None}

        # 2. Ignore tiny drift (< 2m) when standing still
        if self.last_pos and self.get_distance(self.last_pos[0], self.last_pos[1], lat, lon) < 2.0:
            return {"status": "stationary", "speech": None}
        self.last_pos = (lat, lon)

        current_step = self.steps[self.step_index]
        dist = self.get_distance(lat, lon, current_step["lat"], current_step["lon"])
        self.min_dist = min(self.min_dist, dist)

        # 3. Auto-reroute if user walks 35m away from the turn
        if (dist - self.min_dist) > self.off_route_limit:
            reroute = self.start_route(lat, lon, self.active_dest)
            return {
                "status": "rerouting",
                "distance_to_turn": round(dist, 1),
                "speech": f"Off route. {reroute.get('speech', 'Recalculating.')}"
            }

        # 4. Trigger next turn instruction within 12 meters
        if dist <= self.turn_radius:
            self.step_index += 1
            self.min_dist = float("inf")
            self.halfway_spoken = False

            if self.step_index >= len(self.steps):
                self.steps = []
                return {
                    "status": "arrived",
                    "distance_to_turn": 0.0,
                    "speech": f"You have arrived at {self.active_dest}."
                }

            next_inst = self.steps[self.step_index]["instruction"]
            return {
                "status": "turn_now",
                "distance_to_turn": round(dist, 1),
                "speech": next_inst
            }

        # 5. Halfway check-in on long straight roads (> 60m)
        if current_step["distance"] > 60 and not self.halfway_spoken and dist <= 40:
            self.halfway_spoken = True
            return {
                "status": "walking",
                "distance_to_turn": round(dist, 1),
                "speech": f"Continue straight for {int(dist)} meters."
            }

        return {"status": "walking", "distance_to_turn": round(dist, 1), "speech": None}