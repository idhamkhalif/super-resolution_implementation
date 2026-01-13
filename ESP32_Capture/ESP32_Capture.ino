#include <WebServer.h>
#include <WiFi.h>
#include "esp32cam.h"

const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

WebServer server(80);

// ===== ONLY ONE RESOLUTION =====
static auto hiRes = esp32cam::Resolution::find(800, 600);

void wifi_status() {
  while (WiFi.status() != WL_CONNECTED) {
    digitalWrite(4, HIGH);
    delay(500);
    digitalWrite(4, LOW);
  }
}

void serveJpg() {
  auto frame = esp32cam::capture();
  if (frame == nullptr) {
    Serial.println("CAPTURE FAIL");
    server.send(503, "", "");
    return;
  }

  Serial.printf(
    "CAPTURE OK %dx%d %db\n",
    frame->getWidth(),
    frame->getHeight(),
    static_cast<int>(frame->size())
  );

  server.setContentLength(frame->size());
  server.send(200, "image/jpeg");

  WiFiClient client = server.client();
  frame->writeTo(client);
}

// ===== SINGLE ENDPOINT =====
void handleJpg() {
  serveJpg();
}

void setup() {
  Serial.begin(115200);
  pinMode(4, OUTPUT);
  Serial.println();

  // ===== CAMERA CONFIG =====
  {
    using namespace esp32cam;
    Config cfg;
    cfg.setPins(pins::AiThinker);
    cfg.setResolution(hiRes);   // FIXED SVGA
    cfg.setBufferCount(2);
    cfg.setJpeg(80);

    bool ok = Camera.begin(cfg);
    Serial.println(ok ? "CAMERA OK" : "CAMERA FAIL");
  }

  // ===== WIFI =====
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  wifi_status();

  Serial.print("📸 Snapshot URL: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/image-svga.jpg");

  // ===== WEB SERVER =====
  server.on("/image-svga.jpg", handleJpg);
  server.begin();
}

void loop() {
  server.handleClient();
}
