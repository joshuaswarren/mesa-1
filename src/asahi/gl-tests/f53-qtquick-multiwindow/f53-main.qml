import QtQuick
import QtQuick.Shapes
import QtQuick.Window

Item {
    id: controller
    objectName: "f53Controller"

    property bool running: false
    property bool settled: false
    property int phase: 0
    property int cycleCount: 12

    readonly property string atlasCorpus:
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz 0123456789 "
        + "!@#$%^&*()_+-=[]{};:,.<>/? ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    function growingText(windowIndex, cardIndex) {
        const count = Math.min(atlasCorpus.length,
                               12 + phase * 9 + windowIndex * 5 + cardIndex * 3)
        return atlasCorpus.substring(0, count)
    }

    component NotificationWindow: Window {
        id: notificationWindow

        required property int windowIndex
        property color accent: ["#66ccff", "#ff66cc", "#f4df6b"][windowIndex]

        objectName: "f53Window" + windowIndex
        title: "F53 Qt Quick window " + windowIndex
        flags: Qt.FramelessWindowHint
        color: "transparent"
        visible: controller.running
                 && (controller.settled || ((controller.phase + windowIndex) % 4 !== 1))
        width: controller.settled ? 480 : 448 + ((controller.phase + windowIndex) % 3) * 16
        height: controller.settled ? 352 : 328 + ((controller.phase + windowIndex * 2) % 3) * 12
        x: 32 + windowIndex * 36
        y: 32 + windowIndex * 32

        Rectangle {
            anchors.fill: parent
            radius: 22
            color: "#ed090e1b"
            border.width: 1
            border.color: "#8054617a"
            clip: true

            Column {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 7

                Repeater {
                    model: 4

                    delegate: Rectangle {
                        id: card
                        required property int index
                        property color cardAccent: notificationWindow.accent

                        width: parent.width
                        height: (parent.height - 21) / 4
                        radius: 14
                        color: index % 2 ? "#dc171d2d" : "#e21d2639"
                        clip: true

                        Shape {
                            anchors.fill: parent
                            preferredRendererType: Shape.CurveRenderer

                            ShapePath {
                                strokeWidth: 2
                                strokeColor: card.cardAccent
                                fillColor: "transparent"
                                capStyle: ShapePath.RoundCap
                                joinStyle: ShapePath.RoundJoin

                                PathSvg {
                                    path: "M 14 1 H " + (card.width - 14)
                                          + " Q " + (card.width - 1) + " 1 "
                                          + (card.width - 1) + " 14 V "
                                          + (card.height - 14) + " Q "
                                          + (card.width - 1) + " " + (card.height - 1)
                                          + " " + (card.width - 14) + " " + (card.height - 1)
                                          + " H 14 Q 1 " + (card.height - 1)
                                          + " 1 " + (card.height - 14)
                                          + " V 14 Q 1 1 14 1 Z"
                                }
                            }
                        }

                        Image {
                            id: icon
                            property int requestedIconSize: 32 + ((controller.phase + card.index) % 4) * 8
                            objectName: "f53Image-" + notificationWindow.windowIndex + "-" + card.index
                            x: 10
                            anchors.verticalCenter: parent.verticalCenter
                            width: 48
                            height: 48
                            source: Qt.resolvedUrl("f53-oracle-icon.svg")
                            sourceSize: Qt.size(requestedIconSize, requestedIconSize)
                            asynchronous: true
                            cache: true
                            smooth: true
                            mipmap: false
                        }

                        Text {
                            objectName: "f53NativeText-" + notificationWindow.windowIndex + "-" + card.index
                            x: 50
                            y: 5
                            width: 26
                            text: "N" + card.index
                            color: card.cardAccent
                            font.pixelSize: 11 + ((notificationWindow.windowIndex + card.index) % 3)
                            font.bold: true
                            renderType: Text.NativeRendering
                        }

                        Text {
                            objectName: "f53SdfText-" + notificationWindow.windowIndex + "-" + card.index + "-title"
                            x: 70
                            y: 8
                            width: parent.width - 84
                            text: "DISTANCE FIELD " + (notificationWindow.windowIndex + 1)
                                  + "." + (card.index + 1) + " / cycle " + controller.phase
                            color: "#fff7ee"
                            font.pixelSize: 13 + ((notificationWindow.windowIndex + card.index) % 3)
                            font.weight: card.index % 2 ? Font.DemiBold : Font.Medium
                            elide: Text.ElideRight
                        }

                        Text {
                            objectName: "f53SdfText-" + notificationWindow.windowIndex + "-" + card.index + "-body"
                            x: 70
                            y: 29
                            width: parent.width - 84
                            height: parent.height - 34
                            text: controller.growingText(notificationWindow.windowIndex, card.index)
                            color: "#cbd7ea"
                            font.pixelSize: 9 + ((controller.phase + card.index) % 3)
                            wrapMode: Text.WrapAnywhere
                            maximumLineCount: 2
                            elide: Text.ElideRight
                        }
                    }
                }
            }
        }
    }

    NotificationWindow { windowIndex: 0 }
    NotificationWindow { windowIndex: 1 }
    NotificationWindow { windowIndex: 2 }
}
