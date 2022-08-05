import requests
import hoboJUKI_IDs


class hoboJUKIdmLogger:
    def __init__(self,webhook_url):
        self.webhook_url = webhook_url

    def send_message(self, message,author_name):
        send_json = {"content": f"[{author_name}]<s>{message}</s>"}
        response = requests.post(self.webhook_url, json=send_json)

if __name__ == "__main__":
    logger = hoboJUKIdmLogger(hoboJUKI_IDs.webhook_url)
    logger.send_message("hello","juki")