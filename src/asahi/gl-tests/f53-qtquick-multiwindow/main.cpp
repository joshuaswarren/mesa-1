#include <QCommandLineParser>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QGuiApplication>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QQmlApplicationEngine>
#include <QQuickItem>
#include <QQuickWindow>
#include <QSet>
#include <QTimer>
#include <QVersionNumber>

#include <algorithm>
#include <atomic>
#include <memory>

namespace {

struct WindowState {
    QQuickWindow *window = nullptr;
    std::atomic<int> sceneGraphInitializations{0};
    std::atomic<int> frames{0};
    QImage reference;
};

QString imageHash(const QImage &image)
{
    QCryptographicHash hash(QCryptographicHash::Sha256);
    for (int y = 0; y < image.height(); ++y)
        hash.addData(QByteArrayView(reinterpret_cast<const char *>(image.constScanLine(y)),
                                    image.bytesPerLine()));
    return QString::fromLatin1(hash.result().toHex());
}

QJsonObject inspectImage(const QImage &source, const QSize &expectedSize, QString *failure)
{
    const QImage image = source.convertToFormat(QImage::Format_RGBA8888);
    QSet<QRgb> colors;
    qsizetype transparent = 0;
    qsizetype dark = 0;
    qsizetype bright = 0;
    qsizetype saturated = 0;

    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const QColor c = image.pixelColor(x, y);
            colors.insert(c.rgba());
            transparent += c.alpha() < 16;
            dark += c.alpha() > 180 && c.red() < 70 && c.green() < 80 && c.blue() < 100;
            bright += c.alpha() > 180 && c.red() > 175 && c.green() > 175 && c.blue() > 165;
            const int maximum = std::max({c.red(), c.green(), c.blue()});
            const int minimum = std::min({c.red(), c.green(), c.blue()});
            saturated += c.alpha() > 160 && maximum > 145 && maximum - minimum > 65;
        }
    }

    const qsizetype pixels = qsizetype(image.width()) * image.height();
    const bool cornersTransparent = image.pixelColor(0, 0).alpha() < 32
        && image.pixelColor(image.width() - 1, 0).alpha() < 32
        && image.pixelColor(0, image.height() - 1).alpha() < 32
        && image.pixelColor(image.width() - 1, image.height() - 1).alpha() < 32;

    if (image.size() != expectedSize)
        *failure = QStringLiteral("unexpected settled image size %1x%2")
                       .arg(image.width()).arg(image.height());
    else if (!cornersTransparent)
        *failure = QStringLiteral("rounded-window transparent corners are missing");
    else if (transparent < pixels / 500)
        *failure = QStringLiteral("too few transparent pixels: %1").arg(transparent);
    else if (dark < pixels / 2)
        *failure = QStringLiteral("translucent dark card coverage is too small: %1").arg(dark);
    else if (bright < 250)
        *failure = QStringLiteral("bright glyph coverage is too small: %1").arg(bright);
    else if (saturated < 350)
        *failure = QStringLiteral("icon/border accent coverage is too small: %1").arg(saturated);
    else if (colors.size() < 96)
        *failure = QStringLiteral("image color diversity is too small: %1").arg(colors.size());

    return QJsonObject{{QStringLiteral("width"), image.width()},
                       {QStringLiteral("height"), image.height()},
                       {QStringLiteral("transparentPixels"), qint64(transparent)},
                       {QStringLiteral("darkPixels"), qint64(dark)},
                       {QStringLiteral("brightPixels"), qint64(bright)},
                       {QStringLiteral("saturatedPixels"), qint64(saturated)},
                       {QStringLiteral("uniqueColors"), colors.size()},
                       {QStringLiteral("sha256"), imageHash(image)}};
}

void fail(const QString &message)
{
    qCritical().noquote() << "F53 FAIL:" << message;
    QCoreApplication::exit(1);
}

bool auditScene(const QList<std::shared_ptr<WindowState>> &states, QString *failure)
{
    int nativeTexts = 0;
    int sdfTexts = 0;
    int nativeRenderType = -1;
    int sdfRenderType = -1;
    int images = 0;

    for (const auto &state : states) {
        if (state->window->rendererInterface()->graphicsApi()
            != QSGRendererInterface::OpenGL) {
            *failure = state->window->objectName() + QStringLiteral(" did not initialize OpenGL RHI");
            return false;
        }
        QList<QObject *> sceneObjects;
        QList<QQuickItem *> pending{state->window->contentItem()};
        while (!pending.isEmpty()) {
            QQuickItem *item = pending.takeLast();
            sceneObjects.append(item);
            pending.append(item->childItems());
        }
        for (QObject *object : sceneObjects) {
            if (object->objectName().startsWith(QStringLiteral("f53NativeText"))) {
                ++nativeTexts;
                const int value = object->property("renderType").toInt();
                nativeRenderType = nativeRenderType < 0 ? value : nativeRenderType;
                if (value != nativeRenderType) {
                    *failure = QStringLiteral("native Text renderType mismatch");
                    return false;
                }
            } else if (object->objectName().startsWith(QStringLiteral("f53SdfText"))) {
                ++sdfTexts;
                const int value = object->property("renderType").toInt();
                sdfRenderType = sdfRenderType < 0 ? value : sdfRenderType;
                if (value != sdfRenderType) {
                    *failure = QStringLiteral("default Text renderType mismatch");
                    return false;
                }
            } else if (object->objectName().startsWith(QStringLiteral("f53Image"))) {
                ++images;
                if (object->property("status").toInt() != 1) {
                    *failure = object->objectName() + QStringLiteral(" was not Image.Ready");
                    return false;
                }
            }
        }
    }
    if (nativeTexts != 12 || sdfTexts != 24 || images != 12
        || nativeRenderType < 0 || sdfRenderType < 0 || nativeRenderType == sdfRenderType) {
        *failure = QStringLiteral("scene objects/renderers mismatch: native=%1(%2) sdf=%3(%4) images=%5")
                       .arg(nativeTexts).arg(nativeRenderType).arg(sdfTexts).arg(sdfRenderType)
                       .arg(images);
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    QQuickWindow::setGraphicsApi(QSGRendererInterface::OpenGL);
    QGuiApplication app(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("f53-qtquick-multiwindow"));

    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addOption({QStringLiteral("qml"), QStringLiteral("QML fixture path"),
                      QStringLiteral("path")});
    parser.addOption({QStringLiteral("output"), QStringLiteral("Fresh screenshot directory"),
                      QStringLiteral("directory")});
    parser.addOption({QStringLiteral("no-pixels"),
                      QStringLiteral("Validate submission/lifecycle only (for no-op DRM shim)")});
    parser.addOption({QStringLiteral("cycles"), QStringLiteral("Bounded lifecycle cycle count"),
                      QStringLiteral("count"), QStringLiteral("12")});
    parser.addOption({QStringLiteral("timeout-ms"), QStringLiteral("Internal hard deadline"),
                      QStringLiteral("milliseconds"), QStringLiteral("12000")});
    parser.process(app);

    const bool noPixels = parser.isSet(QStringLiteral("no-pixels"));
    const QString qmlPath = QFileInfo(parser.value(QStringLiteral("qml"))).absoluteFilePath();
    const QString outputPath = QFileInfo(parser.value(QStringLiteral("output"))).absoluteFilePath();
    bool ok = false;
    const int cycles = parser.value(QStringLiteral("cycles")).toInt(&ok);
    if (!ok || cycles < 4 || cycles > 24) {
        qCritical() << "--cycles must be in [4, 24]";
        return 2;
    }
    const int timeoutMs = parser.value(QStringLiteral("timeout-ms")).toInt(&ok);
    if (!ok || timeoutMs < 2000 || timeoutMs > 30000) {
        qCritical() << "--timeout-ms must be in [2000, 30000]";
        return 2;
    }
    if (!QFileInfo::exists(qmlPath)) {
        qCritical().noquote() << "missing QML fixture:" << qmlPath;
        return 2;
    }
    if (!noPixels && (outputPath.isEmpty() || !QDir(outputPath).exists())) {
        qCritical().noquote() << "--output must name an existing fresh directory";
        return 2;
    }

    const QVersionNumber runtime = QVersionNumber::fromString(QString::fromLatin1(qVersion()));
    if (runtime < QVersionNumber(6, 11) || runtime >= QVersionNumber(6, 12)) {
        qCritical().noquote() << "F53 requires Qt 6.11.x; found" << qVersion();
        return 2;
    }
    if (QCoreApplication::testAttribute(Qt::AA_ShareOpenGLContexts)) {
        qCritical() << "F53 requires unshared per-window OpenGL contexts";
        return 2;
    }

    QQmlApplicationEngine engine;
    engine.load(QUrl::fromLocalFile(qmlPath));
    if (engine.rootObjects().size() != 1) {
        qCritical() << "expected one QML controller root";
        return 1;
    }
    QObject *controller = engine.rootObjects().constFirst();
    controller->setProperty("cycleCount", cycles);

    QList<QQuickWindow *> windows;
    for (QWindow *candidate : QGuiApplication::allWindows()) {
        if (auto *window = qobject_cast<QQuickWindow *>(candidate);
            window && window->objectName().startsWith(QStringLiteral("f53Window")))
            windows.append(window);
    }
    std::sort(windows.begin(), windows.end(), [](const QQuickWindow *a, const QQuickWindow *b) {
        return a->objectName() < b->objectName();
    });
    if (windows.size() != 3) {
        qCritical() << "expected exactly three F53 QQuickWindows; found" << windows.size();
        return 1;
    }

    auto states = std::make_shared<QList<std::shared_ptr<WindowState>>>();
    for (QQuickWindow *window : windows) {
        auto state = std::make_shared<WindowState>();
        state->window = window;
        QObject::connect(window, &QQuickWindow::sceneGraphInitialized, window,
                         [state] { ++state->sceneGraphInitializations; },
                         Qt::DirectConnection);
        QObject::connect(window, &QQuickWindow::frameSwapped, window,
                         [state] { ++state->frames; }, Qt::DirectConnection);
        states->append(state);
    }

    QTimer deadline;
    deadline.setSingleShot(true);
    QObject::connect(&deadline, &QTimer::timeout, &app, [] {
        fail(QStringLiteral("internal deadline expired"));
    });
    deadline.start(timeoutMs);

    auto phase = std::make_shared<int>(0);
    auto lifecycle = std::make_shared<QTimer>();
    lifecycle->setInterval(70);
    QObject::connect(lifecycle.get(), &QTimer::timeout, &app,
                     [controller, cycles, phase, lifecycle, states, noPixels, outputPath] {
        ++*phase;
        controller->setProperty("phase", *phase);
        if (*phase < cycles)
            return;

        lifecycle->stop();
        controller->setProperty("settled", true);
        controller->setProperty("phase", cycles);
        for (const auto &state : *states) {
            state->window->show();
            state->window->requestUpdate();
        }

        QTimer::singleShot(900, qApp, [states, noPixels, outputPath, cycles] {
            QString sceneFailure;
            if (!auditScene(*states, &sceneFailure)) {
                fail(sceneFailure);
                return;
            }
            for (const auto &state : *states) {
                if (state->sceneGraphInitializations.load() < 1 || state->frames.load() < 2) {
                    fail(QStringLiteral("%1 lifecycle incomplete: sceneGraphs=%2 frames=%3 visible=%4 exposed=%5")
                             .arg(state->window->objectName())
                             .arg(state->sceneGraphInitializations.load())
                             .arg(state->frames.load())
                             .arg(state->window->isVisible())
                             .arg(state->window->isExposed()));
                    return;
                }
                if (!noPixels)
                    state->reference = state->window->grabWindow().convertToFormat(QImage::Format_RGBA8888);
            }

            if (noPixels) {
                qInfo().noquote() << "F53 PASS: Qt 6.11 three-window lifecycle/submission; pixels skipped";
                QCoreApplication::exit(0);
                return;
            }

            QTimer::singleShot(140, qApp, [states, outputPath, cycles] {
                QJsonArray reports;
                QSet<QString> hashes;
                for (const auto &state : *states) {
                    const QImage second = state->window->grabWindow().convertToFormat(QImage::Format_RGBA8888);
                    if (state->reference != second) {
                        fail(state->window->objectName()
                             + QStringLiteral(" changed between two settled exact captures"));
                        return;
                    }
                    QString failure;
                    const qreal dpr = state->window->devicePixelRatio();
                    const QSize expectedSize(qRound(state->window->width() * dpr),
                                             qRound(state->window->height() * dpr));
                    QJsonObject report = inspectImage(second, expectedSize, &failure);
                    if (!failure.isEmpty()) {
                        fail(state->window->objectName() + QStringLiteral(": ") + failure);
                        return;
                    }
                    const QString fileName = state->window->objectName() + QStringLiteral(".png");
                    if (!second.save(QDir(outputPath).filePath(fileName))) {
                        fail(QStringLiteral("could not save ") + fileName);
                        return;
                    }
                    report.insert(QStringLiteral("window"), state->window->objectName());
                    report.insert(QStringLiteral("frames"), state->frames.load());
                    report.insert(QStringLiteral("sceneGraphInitializations"),
                                  state->sceneGraphInitializations.load());
                    hashes.insert(report.value(QStringLiteral("sha256")).toString());
                    reports.append(report);
                }
                if (hashes.size() != 3) {
                    fail(QStringLiteral("the three window captures are not structurally distinct"));
                    return;
                }

                const QJsonObject manifest{
                    {QStringLiteral("qtVersion"), QString::fromLatin1(qVersion())},
                    {QStringLiteral("graphicsApi"), QStringLiteral("OpenGL")},
                    {QStringLiteral("cycles"), cycles},
                    {QStringLiteral("windows"), reports},
                };
                QFile manifestFile(QDir(outputPath).filePath(QStringLiteral("manifest.json")));
                if (!manifestFile.open(QIODevice::WriteOnly | QIODevice::NewOnly)
                    || manifestFile.write(QJsonDocument(manifest).toJson()) < 0) {
                    fail(QStringLiteral("could not create fresh manifest.json"));
                    return;
                }
                qInfo().noquote() << "F53 PASS: Qt 6.11 three-window exact/structural pixels; output"
                                  << outputPath;
                QCoreApplication::exit(0);
            });
        });
    });

    controller->setProperty("running", true);
    for (const auto &state : *states)
        state->window->show();
    lifecycle->start();
    return app.exec();
}
