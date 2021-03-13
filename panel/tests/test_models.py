import time

from django.test import TestCase

from panel import tasks
from panel.models import MPackage, Dedic, Server
from panel.tasks import ServerUnit, DedicUnit


class ModelsTestCase(TestCase):

    def test_servers_create(self):
        mpackage = MPackage.objects.create(
            name="MPackage test",
            master="packages/Master.zip"
        )
        mpackage.save()

        dedic = Dedic.objects.create(
            name="Dedic Master test",
            ip="5.180.138.187",
            user_root="root",
            user_single="unittest1",
            password_root="CHBE644Q7x82",
            ssh_key=False
        )
        dedic.save()

        d = DedicUnit(dedic)
        d.init()

        master = Server.objects.create(
            dedic=dedic,
            name="Master test",
            package=mpackage
        )
        master.save()

        s = ServerUnit(master)
        s.init()
        
        self.assertEqual(master.get_last_status(), None)

        s.delete()
        d.delete()

