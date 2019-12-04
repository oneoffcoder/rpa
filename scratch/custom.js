$(document).ready(function(){ 
    console.log('ready');
    $('#play').click(function() {
        console.log('play clicked');
        const syllables = $('#rpa')
            .val()
            .split(' ')
            .filter(item => item.length > 0);

        if (syllables.length < 1) {
            return;
        } else {
            console.log('got text');
            const sounds = syllables
                .map((item) => `mp3/${item}.mp3`)
                .map((item) => new Audio(item));
            
            let i = -1;
            play();
            function play() {
                i++;
                if (i == sounds.length) {
                    return;
                }
                sounds[i].addEventListener('ended', play);
                sounds[i].play();
            }
        }
    });
    console.log('done');
});