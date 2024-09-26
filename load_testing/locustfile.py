from locust import HttpUser, task, between

class LoadTestUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def send_image(self):
        image_url = "https://cf.ltkcdn.net/family/images/orig/200821-2121x1414-family.jpg"
        self.client.post("/analyze", json={"image_url": image_url})
