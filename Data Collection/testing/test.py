from locust import HttpUser, task

class WebsiteUser(HttpUser):
    @task
    def load_homepage(self):
        self.client.get("/")