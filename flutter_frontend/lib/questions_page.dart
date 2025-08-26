import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:path/path.dart' as p;
import 'package:audioplayers/audioplayers.dart';
import 'package:camera/camera.dart';
import 'dart:async';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';


class Quiz extends StatefulWidget {
  final String username;
  const Quiz({super.key,required this.username});
  State<Quiz> createState() {
    return _QuizState();
  }
}
class ToneResult {
  final String prediction;
  final double confidence;
  final Map<String, double> allProbs;

  ToneResult({
    required this.prediction,
    required this.confidence,
    required this.allProbs,
  });

  factory ToneResult.fromJson(Map<String, dynamic> json) {
    final probs = (json['all_probs'] as Map<String, dynamic>).map(
      (k, v) => MapEntry(k, (v as num).toDouble()),
    );
    return ToneResult(
      prediction: json['prediction'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      allProbs: probs,
    );
  }
}

class _QuizState extends State<Quiz> {
  bool isRecording = false;
  String? recordingPath;
  String? _transcriptionText;
  ToneResult? _tone;
  Timer? periodicTimer;
 //final audioRecorder = AudioRecorder();
  final AudioRecorder audioRecorder = AudioRecorder();
  
  var activeScreen = 'previous-screen';
  switchScreen() {
    setState(() {
      activeScreen = 'questions-screen';
    });
  }
Future<void> sendImage(File imageFile) async {
  final request = http.MultipartRequest(
    'POST',
    Uri.parse('http://127.0.0.1:8000/predict-emotion'),
  );

  request.files.add(
    await http.MultipartFile.fromPath('file', imageFile.path),
  );

  final response = await request.send();
  final responseBody = await response.stream.bytesToString();

  print("Emotion prediction: $responseBody"); 
}

void startCapturingPeriodically() {
  periodicTimer = Timer.periodic(Duration(seconds: 3), (timer) async {
    try {
      final image = await cameraController!.takePicture();

      print('Captured at: ${image.path}');

      // Now you can send this image to your backend:
      await sendImage(File(image.path));

    } catch (e) {
      print('Error capturing image: $e');
    }
  });
}
// Function to upload the recording to FastAPI
  Future<String?> uploadRecording(String filePath, String username) async {
    final uri = Uri.parse("http://127.0.0.1:8000/upload-audio/");
    final request = http.MultipartRequest('POST', uri);
    
    request.files.add(await http.MultipartFile.fromPath('file', filePath));
    request.fields['username'] = username;

    final response = await request.send();
 
    if (response.statusCode == 200) {
      final responseBody = await response.stream.bytesToString();
      final decoded = json.decode(responseBody);
      return decoded["transcript"]; // get transcript from FastAPI
    } else {
      print("Upload failed with status: ${response.statusCode}");
      return null;
    }
  }

  Future<ToneResult?> predictTone(String filePath) async {
    final uri = Uri.parse("http://127.0.0.1:8000/predict-tone");
    final req = http.MultipartRequest('POST', uri)
      ..files.add(
        await http.MultipartFile.fromPath(
          'file',
          filePath,
          contentType: MediaType('audio', 'wav'), // change if mp3/m4a
        ),
      );

    final res = await req.send();
    final body = await res.stream.bytesToString();
    if (res.statusCode == 200) {
      final decoded = json.decode(body) as Map<String, dynamic>;
      return ToneResult.fromJson(decoded);
    } else {
      print("predict-tone failed: ${res.statusCode} - $body");
      return null;
    }
  }


  void toggleRecording() async {
  try {
    if (!isRecording) {
      await _setupCameraController();

      if (await audioRecorder.hasPermission()) {
        startCapturingPeriodically();

        final Directory appDocumentsDir =
            await getApplicationDocumentsDirectory();
        final String filePath = p.join(appDocumentsDir.path, "recording.wav");

        await audioRecorder.start(
          const RecordConfig(
              // encoder: AudioEncoder.pcm16bits,
              // sampleRate: 16000,
              // numChannels: 1,
              ),
          path: filePath,
        );

        setState(() {
          isRecording = true;
          recordingPath = null;
        });
      }
    } else {
      // Stop both audio and image capture
      periodicTimer?.cancel();
      print("Stopped periodic capture.");

      String? filePath = await audioRecorder.stop();
      await Future.delayed(const Duration(milliseconds: 500));

      if (filePath != null) {
        setState(() {
          isRecording = false;
          recordingPath = filePath;
        });

        final results = await Future.wait([
            uploadRecording(filePath, widget.username), // transcript
            predictTone(filePath),                       // tone
          ]);
           final transcript = results[0] as String?;
          final tone = results[1] as ToneResult?;
          setState(() {
            if (transcript != null) _transcriptionText = transcript;
            if (tone != null) _tone = tone;
          });

          print("Recording saved at: $filePath");
          print("Transcript: $_transcriptionText");
          print("Tone: ${_tone?.prediction} (${_tone?.confidence})");
          print("Tone probs: ${tone?.allProbs}");
      }
    }
  } catch (e) {
    print("Error during recording toggle: $e");
    setState(() {
      isRecording = false;
    });
  }
}
  AudioPlayer? _audioPlayer;
  @override
  void dispose() {
    _audioPlayer?.dispose();
    super.dispose();
  }

  @override
  void initState() {
    super.initState();
    _setupCameraController();
  }

  List<CameraDescription> cameras = [];
  CameraController? cameraController;
Future<void> _setupCameraController() async {
  final _cameras = await availableCameras();
  final frontCamera = _cameras.firstWhere(
    (camera) => camera.lensDirection == CameraLensDirection.front,
    orElse: () => _cameras.first,
  );

  cameraController = CameraController(
    frontCamera,
    ResolutionPreset.medium,
  );

  await cameraController!.initialize();
  setState(() {});
}
  Widget _buildUI() {
    if (cameraController == null ||
        cameraController?.value.isInitialized == false) {
      return const Center(child: CircularProgressIndicator());
    }
    return SafeArea(

      child: SizedBox(
        
        //width:screenWidth*0.8
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            const SizedBox(height: 160),
            Transform.translate(
            offset: const Offset(0, -90),
             child: Container(
               padding: EdgeInsets.symmetric(horizontal: 20,),
               child: SizedBox(
                width: MediaQuery.of(context).size.width * 0.9,
                height: MediaQuery.of(context).size.height*0.4,
                 child: AspectRatio(
                             aspectRatio: cameraController!.value.aspectRatio,
                 child: CameraPreview(cameraController!),
                        ),
               ),
             ),
       
           ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Stack(
        children: [
          _buildUI(),

          Center(
            child: Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: 16.0,
                vertical: 100,
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  // Top row
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      IconButton(
                      onPressed: () => Navigator.pushNamed(context, '/questions_screen'), 
                      icon: const Icon(Icons.arrow_back),
                      iconSize: 28,
                      ),
                      
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 3,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Text(
                          'Question 3',
                          style: TextStyle(
                            fontSize: 25,
                            color: Colors.black,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                      IconButton(
                      onPressed: () => Navigator.pushNamed(context, '/login_page'), 
                      icon: const Icon(Icons.arrow_forward),
                      iconSize: 28,
                      ),
                    ],
                  ),
                  const SizedBox(height: 5),

                  // Question bubble
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      //color: const Color.fromARGB(255, 91, 106, 234),
                      color:const Color(0xFFEDEDED),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: const Text(
                      "You're leading a project, and a key team member unexpectedly takes leave, causing delays. How would you handle the situation to keep the project on track?",
                      style: TextStyle(
                        color: Colors.black,
                        fontSize: 20,
                        height: 1.4,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ),

                  const SizedBox(height: 360),

                  // Recording button
                  ElevatedButton.icon(
                    onPressed: toggleRecording,
                    icon: Icon(isRecording ? Icons.stop : Icons.mic),
                    label: Text(
                      isRecording ? 'Stop Recording' : 'Start Recording',
                    ),
                    style: ElevatedButton.styleFrom(
                      foregroundColor: Colors.black,
                      backgroundColor: Color(0xFFBEE9EC),
                      textStyle: const TextStyle(fontSize: 20),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 24,
                        vertical: 12,
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),

                  const SizedBox(height: 30),
                  if (recordingPath != null ) ...[
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        ElevatedButton.icon(
                          onPressed: () async {
                            
                            _audioPlayer ??= AudioPlayer();
                           if (recordingPath != null && await File(recordingPath!).exists()) {
                            await _audioPlayer!.play(
                              DeviceFileSource(recordingPath!),

                            );
                            _audioPlayer!.onPlayerComplete.listen((event) {
                              print("Playback complete");
                           });
                            // Optionally print duration
                            await Future.delayed(Duration(milliseconds: 300));
                            final duration = await _audioPlayer!.getDuration();
                            print(
                              "Audio duration: ${duration?.inMilliseconds ?? 'unknown'} ms",
                            );
                           }
                           else{
                            print("\n\n\nFile does not exist at: $recordingPath \n\n\n\n");
                           } 
                          },
                          icon: Icon(Icons.play_arrow, color: Colors.black54),
                          label: Text('Play Recording'),
                          style: ElevatedButton.styleFrom(
                            foregroundColor: Colors.black,
                            backgroundColor: Color(0xFFEDEDED),
                            textStyle: const TextStyle(fontSize: 20),
                            padding: const EdgeInsets.symmetric(
                              vertical: 16,
                              horizontal: 12,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(20),
                            ),
                          ),
                        ),
                        //Transcription 
                        ElevatedButton.icon(
                          onPressed: () {
                            if (_transcriptionText != null) {
                              showDialog(
                                context: context,

                                builder:
                                    (_) => AlertDialog(
                                      
                                      title: const Text(
                                        "Transcription",
                                        textAlign: TextAlign.center,
                                        style: TextStyle(
                                          fontWeight: FontWeight.bold,
                                          color: Colors.black,
                                        ),
                                      ),
                                      content: SingleChildScrollView(
                                        child: Text(
                                          _transcriptionText!,
                                          softWrap: true,
                                          style: const TextStyle(
                                            color: Colors.black,
                                            fontSize: 16,
                                          ),
                                        ),
                                      ),

                                      actions: [
                                        TextButton(
                                          child: const Text("Close"),
                                          onPressed:
                                              () => Navigator.pop(context),
                                        ),
                                      ],
                                    ),
                              );
                            } else {
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                  content: Text("No transcription available."),
                                ),
                              );
                            }
                          },
                          icon: Icon(Icons.visibility, color: Colors.black54),
                          label: Text('See Transcription'),
                          style: ElevatedButton.styleFrom(
                            foregroundColor: Colors.black,
                            backgroundColor: Color(0xFFEDEDED),
                            textStyle: const TextStyle(fontSize: 20),
                            padding: const EdgeInsets.symmetric(
                              vertical: 16,
                              horizontal: 12,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(20),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
