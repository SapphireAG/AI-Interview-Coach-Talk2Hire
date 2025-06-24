import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:path/path.dart' as p;
import 'package:audioplayers/audioplayers.dart';
import 'package:camera/camera.dart';

import 'package:http/http.dart' as http;

class Quiz extends StatefulWidget {
  const Quiz({super.key});
  State<Quiz> createState() {
    return _QuizState();
  }
}

class _QuizState extends State<Quiz> {
  final AudioRecorder audioRecorder = AudioRecorder();
  String? _transcriptionText;
  var activeScreen = 'previous-screen';
  switchScreen() {
    setState(() {
      activeScreen = 'questions-screen';
    });
  }

  Future<String?> uploadRecording(String filePath) async {
    final uri = Uri.parse("http://127.0.0.1:8000/upload-audio/");
    final request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('file', filePath));

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
  // @override
  // Widget build(context){
  //   return MaterialApp(
  //     home: Scaffold(
  //       backgroundColor: Colors.white,
  //       body: Container(
  //         color: Colors.purple,
  //         child:  Column(
  //           crossAxisAlignment: CrossAxisAlignment.center,
  //     children: [
  //       const SizedBox(height: 55),

  //       Center(
  //         child: Container(
  //           // color:Colors.blue,
  //           padding: const EdgeInsets.all(12),
  //           decoration: BoxDecoration(
  //             color: Colors.blue,
  //             borderRadius: BorderRadius.circular(12),
  //           ),
  //           child: const Text(
  //             'Question x',
  //             style:TextStyle(
  //                 color: Colors.white,
  //                 fontSize: 16,
  //               ),

  //             ),

  //           ),

  //       ),
  //     ],
  //         ),

  //     ),
  //       ),
  //   );

  // }
  bool isRecording = false;
  String? recordingPath;
  void toggleRecording() async {
    if (!isRecording) {
      await _setupCameraController();
      if (await audioRecorder.hasPermission()) {
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
      String? filePath = await audioRecorder.stop();
      await Future.delayed(const Duration(milliseconds: 500));
      if (filePath != null) {
        setState(() {
          isRecording = false;
          recordingPath = filePath;
        });

        // Upload the recording
        final transcript = await uploadRecording(filePath);
if (transcript != null) {
  setState(() {
    _transcriptionText = transcript;
  });
  print("Transcript: $transcript");
}

        print("Recording saved at: $filePath");
      }
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
    List<CameraDescription> _cameras = await availableCameras();
    if (_cameras.isNotEmpty) {
      setState(() {
        cameras = _cameras;
        cameraController = CameraController(
          _cameras.first,
          ResolutionPreset.high,
        );
      });
      cameraController?.initialize().then((_) {
        setState(() {});
      });
    }
  }

  Widget _buildUI() {
    if (cameraController == null ||
        cameraController?.value.isInitialized == false) {
      return const Center(child: CircularProgressIndicator());
    }
    return SafeArea(
      child: SizedBox.expand(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            const SizedBox(height: 32),
            Transform.translate(
              offset: const Offset(0, -90),
              child: SizedBox(
                height: MediaQuery.sizeOf(context).height * 0.3,
                width: MediaQuery.sizeOf(context).width * 0.8,
                child: CameraPreview(cameraController!),
              ),
            ),
            // Transform.translate(
            //   offset: const Offset(-4, -30),
            // child: IconButton(
            //   onPressed: (){},
            //   icon: const Icon(
            //        Icons.camera,
            //        color: Colors.red,
            // ),
            // iconSize: 36,
            // ),
            // ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
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
                        const Icon(Icons.arrow_back, size: 28),
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
                        const Icon(Icons.arrow_forward, size: 28),
                      ],
                    ),
                    const SizedBox(height: 5),

                    // Question bubble
                    Container(
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: const Color.fromARGB(255, 91, 106, 234),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: const Text(
                        "You're leading a project, and a key team member unexpectedly takes leave, causing delays. How would you handle the situation to keep the project on track?",
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          height: 1.4,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),

                    const SizedBox(height: 250),

                    // Recording button
                    ElevatedButton.icon(
                      onPressed: toggleRecording,
                      icon: Icon(isRecording ? Icons.stop : Icons.mic),
                      label: Text(
                        isRecording ? 'Stop Recording' : 'Start Recording',
                      ),
                      style: ElevatedButton.styleFrom(
                        foregroundColor: Colors.black,
                        backgroundColor: Colors.white,
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
                    if (recordingPath != null) ...[
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          ElevatedButton.icon(
                            onPressed: () async {
                              _audioPlayer ??= AudioPlayer();

                              await _audioPlayer!.play(
                                DeviceFileSource(recordingPath!),
                              );
                              _audioPlayer!.onPlayerComplete.listen((event) {
                                print("Playback complete");
                              });
                              // Optionally print duration
                              await Future.delayed(Duration(milliseconds: 300));
                              final duration =
                                  await _audioPlayer!.getDuration();
                              print(
                                "Audio duration: ${duration?.inMilliseconds ?? 'unknown'} ms",
                              );
                            },
                            icon: Icon(Icons.play_arrow, color: Colors.white),
                            label: Text('Play Recording'),
                            style: ElevatedButton.styleFrom(
                              foregroundColor: Colors.white,
                              backgroundColor: Color.fromARGB(
                                255,
                                91,
                                106,
                                234,
                              ),
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
      builder: (_) => AlertDialog(
        title: const Text("Transcription"),
        content: Text(_transcriptionText!),
        actions: [
          TextButton(
            child: const Text("Close"),
            onPressed: () => Navigator.pop(context),
          ),
        ],
      ),
    );
  } else {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text("No transcription available.")),
    );
  }
                            },
                            icon: Icon(Icons.visibility, color: Colors.white),
                            label: Text('See Transcription'),
                            style: ElevatedButton.styleFrom(
                              foregroundColor: Colors.white,
                              backgroundColor: Color.fromARGB(
                                255,
                                91,
                                106,
                                234,
                              ),
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
      ),
    );
  }
}
