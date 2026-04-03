from locust import HttpUser, task, between

class HackathonJudge(HttpUser):
    # Simulates the wait time between clicks (1 to 3 seconds)
    wait_time = between(1, 3)

    @task(1)
    def load_ui(self):
        """Simulate a judge loading the HTML dashboard."""
        self.client.get("/")

    @task(3)
    def run_full_simulation(self):
        """Simulate a judge testing the AI agent."""
        # 1. Start a new evaluation
        self.client.post("/reset", json={"task_id": "task_01_easy"})
        
        # 2. Rapidly step through 5 days of the simulation
        for _ in range(5):
            # Adjust the payload if your /step endpoint expects specific JSON
            self.client.post("/step", json={})
            
        # 3. Check the scoreboard
        self.client.get("/grader")