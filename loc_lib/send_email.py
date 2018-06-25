from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
import os
from email.utils import COMMASPACE, formatdate
from email import encoders
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


def send_mail_html(server, fro, to, subject, text, files=[]):
    assert type(server) == dict
    assert type(to) == list
    assert type(files) == list

    msg = MIMEMultipart('related')
    msg['From'] = fro
    msg['Subject'] = subject
    msg['To'] = COMMASPACE.join(to)  # COMMASPACE==', '
    msg['Date'] = formatdate(localtime=True)
    Contents = MIMEText(text, 'html', 'gb2312')
    msg.attach(Contents)
    # msg.attach(MIMEText(text))

    for file in files:
        part = MIMEBase('application', 'octet-stream')  # 'octet-stream': binary data
        part.set_payload(open(file, 'rb').read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(file))
        msg.attach(part)

    import smtplib
    # smtp = smtplib.SMTP()
    # smtp.connect(server['name'],server['port'])
    smtp = smtplib.SMTP_SSL(server['name'])
    smtp.login(server['user'], server['passwd'])
    smtp.sendmail(fro, to, msg.as_string())
    smtp.close()


def send_email(text, to, filepath, subject):
    fro = 'Report<sysadmin@dfctec.com>'
    server = dict()
    server['name'] = 'mail.dfc.sh'
    server['name'] = 'smtp.qq.com'
    server['user'] = 'service@dfc.sh'
    server['passwd'] = 'powerup'

    send_mail_html(server, fro, to, subject, text, filepath)
