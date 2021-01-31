import datetime
import json
from pprint import pprint

from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.utils.timezone import now
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

from panel import tasks
from panel.models import ServerStatus, Server, MPackage, SRPackage
from panel.serializers import ServerStatusSerializer, ServerSerializer, MPackageSerializer, SRPackageSerializer


class ServerView(APIView):
    # Инициализация и получение списка всех серверов

    @staticmethod
    def get(request):
        servers = ServerSerializer(Server.objects.all(), many=True)
        m_packages = MPackageSerializer(MPackage.objects.all(), many=True)
        sr_packages = SRPackageSerializer(SRPackage.objects.all(), many=True)
        return Response({
            "servers": servers.data,
            "m_packages": m_packages.data,
            "sr_packages": sr_packages.data,
        })

    @staticmethod
    def post(request):
        server = request.data
        serializer = ServerSerializer(data=server)
        server_saved = None

        if serializer.is_valid(raise_exception=True):
            server_saved = serializer.save()
            tasks.server_task(server_saved.id, "init")

        return Response({"success": "Сервер '{}' добавлен.".format(server_saved.name)})


@method_decorator(csrf_exempt, name='dispatch')
class ServerInstanceView(APIView):
    # Запуск, остановка и получение данных о сервере

    @staticmethod
    def get(request, pk):
        try:
            server = Server.objects.filter(id=pk)

            try:
                status = ServerStatus.objects.filter(server=server.last(), created_at__gte=(
                        now() - datetime.timedelta(minutes=10))).values().last()
            except IndexError:
                status = None

            return Response({"server": server.values(
                'id', 'ip', 'log', 'name', 'password_root', 'password_single', 'ssh_key', 'user_root',
                'user_single', 'm_package__name', 'sr_package__name', 'm_package__created_at',
                'sr_package__created_at', 'config')[0],
                             "status": status})
        except IndexError:
            return Response({"server": None, "status": None})

    @staticmethod
    def put(request, pk):
        tasks.server_task(pk, "start")
        return Response({"success": "Сервер запущен."})

    @staticmethod
    def delete(request, pk):
        tasks.server_task(pk, "stop")
        return Response({"success": "Сервер остановлен."})


@method_decorator(csrf_exempt, name='dispatch')
class ServerStatusView(APIView):
    # Состояния сервера в моменте времени и информация о нагрузке

    @staticmethod
    def get(request):
        stats = ServerStatus.objects.all()
        serializer = ServerSerializer(stats, many=True)
        return Response({"stats": serializer.data})

    @staticmethod
    def post(request):
        stat = {
            'server': request.data['server'],
            'cpu_usage': request.data['cpu_usage'],
            'ram_usage': request.data['ram_usage'],
            'ram_available': request.data['ram_available'],
            'hdd_usage': request.data['hdd_usage'],
            'hdd_available': request.data['hdd_available'],
            'online': request.data['online']['AllPlayers']
        }
        pprint(stat)
        serializer = ServerStatusSerializer(data=stat)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
        return Response({"success": "Статус сервера обновлён."})



@method_decorator(csrf_exempt, name='dispatch')
class ServerOnlineView(APIView):
    # Тестовый view для проверки связи с серевером

    @staticmethod
    def get(request, pk):
        return Response({"online": tasks.get_online()})


class MPackageView(APIView):
    # Загрузка сборки master
    parser_classes = (MultiPartParser,)

    @staticmethod
    def get(request):
        packages = MPackage.objects.all()
        serializer = MPackageSerializer(packages, many=True)
        return Response({"packages": serializer.data})

    @staticmethod
    def post(request):
        serializer = MPackageSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
        return Response({"success": "Сборка загружена"})


@method_decorator(csrf_exempt, name='dispatch')
class MPackageInstanceView(APIView):
    # Получение информации о сборке

    @staticmethod
    def get(request, pk):
        try:
            return Response({
                "code": True,
                "data": {
                    "package": MPackage.objects.filter(id=pk).values()[0]
                }
            })
        except IndexError:
            return Response({
                "ok": False,
                "error": {
                    "id": 404,
                    "message": "Undefined value"
                }
            })


class SRPackageView(APIView):
    # Загрузка сборки spawner + room
    parser_classes = (MultiPartParser,)

    @staticmethod
    def get(request):
        packages = SRPackage.objects.all()
        serializer = SRPackageSerializer(packages, many=True)
        return Response({"packages": serializer.data})

    @staticmethod
    def post(request):
        serializer = SRPackageSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
        return Response({"success": "Сборка загружена"})


@api_view(('POST',))
@renderer_classes((JSONRenderer,))
def update_config(request, pk):
    server = Server.objects.filter(id=pk)
    if server:
        server[0].config = request.data['config']
        server[0].save()
        tasks.server_task(pk, "update_config")
        return Response({"success": "Конфиг обновлён"})
    else:
        return Response({"error": "Сервер не найден"})
