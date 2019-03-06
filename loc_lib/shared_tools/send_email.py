from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
import os
# python 2.3.*: email.Utils email.Encoders
from email.utils import COMMASPACE, formatdate
from email import encoders
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.encoders import encode_base64
from email.message import EmailMessage
import mimetypes


def send_mail_html(server, fro, to, subject, text, files=[]):
    assert type(server) == dict
    assert type(to) == list
    assert type(files) == list
    # msg = MIMEMultipart('related')
    msg = EmailMessage()
    msg['From'] = fro
    msg['Subject'] = subject
    msg['To'] = COMMASPACE.join(to)  # COMMASPACE==', '
    msg['Date'] = formatdate(localtime=True)
    Contents = MIMEText(text, 'html', 'gb2312')
    msg.add_alternative(Contents)
    # msg.attach(MIMEText(text))

    for file in files:
        fp = open(file, 'rb')

        msgBase = MIMEBase('application', 'octet-stream')  # 'octet-stream': binary data
        msgBase.set_payload(fp.read())
        encoders.encode_base64(msgBase)
        msgBase.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(file))
        msg.attach(msgBase)

        mimetype, encoding = mimetypes.guess_type(file)
        mimetype = mimetype.split('/', 1)
        with open(file, "rb") as fp:
            attachment = MIMEBase(mimetype[0], mimetype[1])
            attachment.set_payload(fp.read())
        encode_base64(attachment)
        attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(file))
        msg.add_attachment(attachment)

    import smtplib
    smtp = smtplib.SMTP_SSL(server['name'])
    smtp.login(server['user'], server['passwd'])
    smtp.sendmail(fro, to, msg.as_string())
    smtp.close()


def send_email(text, to, filepath, subject):
    fro = 'Report<report@yingpei.com>'

    server = dict()
    server['name'] = 'mail.yingpei.com'
    server['user'] = 'report@yingpei.com'
    server['passwd'] = 'Malpha2018MS'

    send_mail_html(server, fro, to, subject, text, filepath)


if __name__ == '__main__':
    send_email('test', ['whs@yingpei.com'], [], 'Wonderfully')
