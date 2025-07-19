import 'dart:async';
import 'package:flutter/material.dart';
import 'questions_screen.dart';
//import 'dart:ui';
// import 'package:flutter_application_2/login_page.dart';
class LandingPage extends StatefulWidget {
  final String username;
  const LandingPage({super.key,required this.username});

  @override
  State<LandingPage> createState() => LandingPageState();
}

class LandingPageState extends State<LandingPage> {
  
  final PageController _pageController = PageController(viewportFraction: 0.8);
  int _currentPage = 0;
  late Timer _autoScrollTimer;

  final List<String> features = [
    'Sentiment Analysis\n\nSaying one thing, sounding like another? Happens! We\'ll help you align tone and intent like a pro.',
    'Facial Expression Analysis\n\nThat raised eyebrow says more than words. Don\'t worry, we\'re decoding your expressions, too.',
    'Targetted Feedback\n\nWords, tone, expressions\nWe watch it all and serve up feedback that actually helps',
    'Transcription\n\nUh-oh moment? We transcribe it all, so you can replay, reflect, and rise stronger.',
    'Personalized Questions\n\nWeak in Machine Learning questions? Dont worry we got your back!',
    'Analytics\n\nClear Indications of your progress and growth areas keeping you motivated to keep improving!',

  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
    _autoScrollTimer = Timer.periodic(Duration(seconds: 3), (Timer timer) {
      if (_pageController.hasClients) {
        int nextpage = (_currentPage + 1) % features.length;  //autoscroll
        //print("Auto-scroll to page $nextpage");
        _pageController.animateToPage(
          nextpage,
          duration: Duration(milliseconds:1400),
          curve: Curves.easeInOut,
        );
      }
    });
    });
  }

  @override
  void dispose() {
    _autoScrollTimer.cancel();
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        toolbarHeight:100,
        title: Padding(
          padding: const EdgeInsets.only(top: 21.0),
          child:Center(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(
              10.0,
            ), 
            // Adjust the radius as needed
            child: Image.asset(
              'assets/NEW_logo.png',
              height: 80.0,
              width: 150.0,
              //scale: 1.0,
              fit: BoxFit.contain,
            ),
          ),
        ),
        ),
        backgroundColor: Color.fromARGB(255, 249, 250, 251),

        leading: IconButton(
          icon: Icon(Icons.menu,color: Colors.black38,size: 35,),
          iconSize: 40,
          color: Colors.white,
          onPressed: () {
            // Handle menu action
          },
        ),

        actions: [
          IconButton(
            icon: Icon(Icons.account_circle),
            iconSize: 40,
            color: Colors.black38,
            onPressed: () {
              // Handle account action
            },
          ),
        ],
      ),
      backgroundColor: const Color(0xFFF9FAFB),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [

              Container(
                padding: EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color:Color.fromARGB(255, 237, 237, 237),
                  borderRadius: BorderRadius.circular(20), 
                ),
                child:  Text(
                  
                  "Hi ${widget.username}, \nWelcome to the start of your new journey!\nFrom Berozgaar to Rozgaar",
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold,fontStyle: FontStyle.italic),
                ),
              ),
              const SizedBox(height: 10),


              Container(
              padding: const EdgeInsets.all(10),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center, 
                children: const [
                  Icon(
                    Icons.star, 
                    color: Colors.black,
                    size: 28,
                  ),
                  SizedBox(width: 8), 
                  Text(
                    "Features:",
                    style: TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                ],
              ),
            ),



              // === Carousel of Features ===
              const SizedBox(height: 20),
              SizedBox(
                height: 260,
                child: PageView.builder( 
                  controller: _pageController,
                  onPageChanged: (int index){
                    _currentPage = index;
                  },
                  itemCount: features.length,
                  itemBuilder: (context, index) {
                    return _buildFeatureCard(features[index],index);
                  },
                ),
              ),
              SizedBox(height: 20,),
              Center(
                child: ElevatedButton.icon(
                onPressed: ()=> Navigator.pushNamed(context,'/questions_screen'),
                label: Text('Get Started!'), 
                icon:Icon(Icons.arrow_forward),
                style: ElevatedButton.styleFrom(
                  foregroundColor: Colors.black,
                  backgroundColor: Color(0xFFBEE9EC) ,
                  padding: const EdgeInsets.symmetric(
                                vertical: 16,
                                horizontal: 12,
                              ),
                shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(20),
                              ),            
                ),
                ),
              ),
              SizedBox(height: 10,),
              //Text('All the Best!')
            ],
          ),
        ),
      ),
    );
  }
  final List<Color> colors= [
    Color(0xFF48b8d0),
    Color(0xFF62c3d7),
    Color(0xFF7bcfde),
    Color(0xFF94dae4),
    Color(0xFFaee5eb),
    Color(0xFFc7f1f2)

  ];
    final List<Image> img= [
     Image.asset('assets/voice-message.png'),
     Image.asset('assets/face_recog.png'),
     Image.asset('assets/feedback.png'),
     Image.asset('assets/transcription.png'),
     Image.asset('assets/personalq.png'),
     Image.asset('assets/analysis.png'),

  ];
  Widget _buildFeatureCard(String text,int index) {
    final parts = text.split('\n\n');
    final title = parts[0];
    final body = parts.length > 1 ? parts[1] : '';

    return Padding(
      padding: const EdgeInsets.only(right: 16),
      child: Container(
        padding: const EdgeInsets.fromLTRB(20,20,20,20),
        decoration: BoxDecoration(
         color: colors[index%colors.length],
         
          borderRadius: BorderRadius.circular(20),
        ),
        
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            children: [
              SizedBox(height:40,
              width:40, 
              child: img[index%img.length]),
              //const SizedBox(height: 2),
              Expanded(
            child: Center(
              child: RichText(
                textAlign: TextAlign.center,
                text: TextSpan(
                  children: [
                    TextSpan(
                      text: "$title\n\n",
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.black,
                      ),
                    ),
                    TextSpan(
                      text: body,
                      style: const TextStyle(
                        fontSize: 16,
                        color: Colors.black,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    ),
  );
}
}