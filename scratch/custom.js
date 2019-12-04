$(document).ready(function(){ 
    console.log('ready');
    $('#play').click(function() {
        console.log('play clicked');
        var sounds = [
            new Audio('mp3/nyob.mp3'),
            new Audio('mp3/zoo.mp3')
        ];

        let i = -1;
        play();
        function play() {
            i++;
            if (i == sounds.length) {
                return;
            }
            sounds[i].addEventListener('ended', play);
            sounds[i].play();
            console.log(sounds[i]);
        }
    });
    console.log('done');
});