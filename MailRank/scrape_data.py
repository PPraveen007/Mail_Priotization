from selenium import webdriver
import time
import pandas as pd

m_Options = webdriver.ChromeOptions()
# m_Options.add_argument("--headless")
# m_Options.add_argument("--no-sandbox")
m_Options.add_argument(
    "--user-data-dir=C:\\Users\\Sonu\\AppData\\Local\\Google\\Chrome\\User Data")
m_Options.add_argument("--profile-directory=Profile 1")
m_Options.add_argument("--disable-extensions")
# 'C:\\Users\\Sonu\\Downloads\\chromedriver_win32\\chromedriver.exe'
driver = webdriver.Chrome(options=m_Options)

driver.get('http://mail.google.com')
first_mail = driver.find_element(by="class name", value="bog")

print("opening first mail...")
first_mail.click()

data = {
    'heading': [],
    'sender_name': [],
    'sender_email': [],
    'content': [],
}

mails_done = 0
while (mails_done < 100):
    try:
        heading = driver.find_element(by="class name", value="hP").text
        sender_name = driver.find_element(by="class name", value="gD").text
        sender_email = driver.find_element(by="class name", value="go").text
        div_element = driver.find_element(
            by="class name", value="a3s")
        content = div_element.text

        # Moving onto next mail
        next_button = driver.find_element(
            by="xpath", value="//div[@aria-label='Older']")
        if (len(heading) == 0):
            print(f"Mail Not loaded. Retrying...", end='')
            time.sleep(0.5)
            print('\b \b'*100, end='')
            continue

        print(f"{heading=}")
        data['heading'].append(heading)
        print(f"{sender_name=}")
        data['sender_name'].append(sender_name)
        print(f"{sender_email=}")
        data['sender_email'].append(sender_email)
        print(f"{div_element=}")
        content = content.replace('\n', ' ')
        content = content.replace('"', ' ')
        content = content.replace('\t', ' ')
        content = content.replace('\r', ' ')
        print(f"{content=}")
        data['content'].append(content)
        print(f"{next_button=}")
        print("Clicking next button")
        # input("Press enter to move to next email")
        next_button.click()
        mails_done += 1
    except:
        print(f"Next button was not found. Retrying...", end='')
        time.sleep(0.5)
        print('\b \b'*100, end='')
        continue
    retry = 100

df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)
print("Done")
input("Press Enter to Quit")
driver.quit()
