"""OpenTelemetry tracing setup and span instrumentation."""
import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from src.core.config import get_settings

logger = logging.getLogger(__name__)
_TRACER_INITIALIZED = False


def init_telemetry() -> trace.Tracer:
    """Initializes the OpenTelemetry TracerProvider with OTLP and Console exporters."""
    global _TRACER_INITIALIZED
    settings = get_settings()

    if _TRACER_INITIALIZED:
        return trace.get_tracer(settings.OTEL_SERVICE_NAME)

    resource = Resource.create({
        "service.name": settings.OTEL_SERVICE_NAME,
        "service.environment": settings.APP_ENV,
    })

    provider = TracerProvider(resource=resource)

    # 1. Console Exporter (for development observability)
    if settings.OTEL_TRACES_CONSOLE_ENABLED and settings.APP_ENV == "development":
        try:
            console_processor = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(console_processor)
        except Exception as e:
            logger.debug("Console exporter disabled: %s", e)

    # 2. OTLP Exporter (for Jaeger in Docker)
    if settings.OTEL_EXPORTER_OTLP_ENDPOINT and settings.OTEL_EXPORTER_OTLP_ENDPOINT.strip():
        try:
            otlp_exporter = OTLPSpanExporter(
                endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT.strip(),
                insecure=True,
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        except Exception as exc:
            logger.debug("OTLP exporter initialization skipped: %s", exc)

    trace.set_tracer_provider(provider)
    _TRACER_INITIALIZED = True
    return trace.get_tracer(settings.OTEL_SERVICE_NAME)


def get_tracer(name: str = "customer-support-agent") -> trace.Tracer:
    """Returns a named tracer instance."""
    init_telemetry()
    return trace.get_tracer(name)
