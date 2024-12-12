import smtplib
from email.mime.text import MIMEText

SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USERNAME = 'garrettstephens11@gmail.com'
SMTP_PASSWORD = 'ybof gyek vtlw bukg'
EMAIL_FROM = 'garrettstephens11@gmail.com'
EMAIL_TO = 'garrettstephens11@gmail.com'
EMAIL_SUBJECT = 'SMTP Configuration Test'

def send_test_email():
    msg = MIMEText('This is a test email to verify SMTP configuration.')
    msg['Subject'] = EMAIL_SUBJECT
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Use SSL/TLS if required
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print(f"Test email sent successfully to {EMAIL_TO}")
    except Exception as e:
        print(f"Failed to send test email: {e}")

if __name__ == '__main__':
    send_test_email()
